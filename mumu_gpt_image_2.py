import base64
import json
import os
import re
import time
import uuid
import http.client
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:
    folder_paths = None


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(NODE_DIR, "config.json")
OMIT = "不发送"
PROTOCOL_OPTIONS = ["自动", "OpenAI同步", "APIMart异步"]
ASPECT_RATIO_OPTIONS = [
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "3:1",
    "1:3",
    "21:9",
    "9:21",
]
RESOLUTION_OPTIONS = ["1k", "2k", "4k", OMIT]
REFERENCE_IMAGE_KEYS = tuple(f"参考图片{i}" for i in range(1, 17))
SIZE_RESOLUTION_MAP = {
    "1:1": {"1k": "1024x1024", "2k": "2048x2048", "4k": "2880x2880"},
    "3:2": {"1k": "1536x1024", "2k": "2048x1360", "4k": "3520x2336"},
    "2:3": {"1k": "1024x1536", "2k": "1360x2048", "4k": "2336x3520"},
    "4:3": {"1k": "1024x768", "2k": "2048x1536", "4k": "3312x2480"},
    "3:4": {"1k": "768x1024", "2k": "1536x2048", "4k": "2480x3312"},
    "5:4": {"1k": "1280x1024", "2k": "2560x2048", "4k": "3216x2576"},
    "4:5": {"1k": "1024x1280", "2k": "2048x2560", "4k": "2576x3216"},
    "16:9": {"1k": "1536x864", "2k": "2048x1152", "4k": "3840x2160"},
    "9:16": {"1k": "864x1536", "2k": "1152x2048", "4k": "2160x3840"},
    "2:1": {"1k": "2048x1024", "2k": "2688x1344", "4k": "3840x1920"},
    "1:2": {"1k": "1024x2048", "2k": "1344x2688", "4k": "1920x3840"},
    "3:1": {"1k": "1881x836", "2k": "3072x1024", "4k": "3840x1280"},
    "1:3": {"1k": "887x1774", "2k": "1024x3072", "4k": "1280x3840"},
    "21:9": {"1k": "2016x864", "2k": "2688x1152", "4k": "3840x1648"},
    "9:21": {"1k": "864x2016", "2k": "1152x2688", "4k": "1648x3840"},
}


def _load_config():
    defaults = {
        "api_base": "https://aimumu.top",
        "api_key": "",
    }
    if not os.path.exists(CONFIG_PATH):
        return defaults
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults.update({k: str(v) for k, v in data.items() if v is not None})
    except Exception:
        pass
    return defaults


def _clean_optional(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in {"", OMIT, "默认"}:
            return None
    return value


def _validate_pixel_size(size):
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        raise ValueError("尺寸必须是 WIDTHxHEIGHT 格式，例如 1536x864")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("尺寸宽高必须大于 0")
    if max(width, height) > 3840:
        raise ValueError("gpt-image-2 尺寸最长边不能超过 3840px")
    if max(width, height) / min(width, height) > 3:
        raise ValueError("gpt-image-2 尺寸长短边比例不能超过 3:1")
    pixels = width * height
    if pixels < 655360 or pixels > 8294400:
        raise ValueError("gpt-image-2 总像素必须在 655,360 到 8,294,400 之间")
    return size


def _resolve_size(selected_size, resolution, custom_size):
    selected_size = _clean_optional(selected_size)
    if selected_size in {"自定义", "自定义像素"}:
        selected_size = _clean_optional(custom_size)
        if selected_size is None:
            raise ValueError("选择自定义像素时，自定义尺寸不能为空")
    if selected_size is None:
        return None
    if selected_size == "auto":
        return selected_size
    resolution = _clean_optional(resolution) or "1k"
    if selected_size in SIZE_RESOLUTION_MAP:
        if resolution not in SIZE_RESOLUTION_MAP[selected_size]:
            raise ValueError("清晰度必须是 1k、2k 或 4k")
        return SIZE_RESOLUTION_MAP[selected_size][resolution]
    return _validate_pixel_size(selected_size)


def _split_image_urls(value):
    value = (value or "").strip()
    if not value:
        return []
    urls = []
    for part in re.split(r"[\r\n,]+", value):
        part = part.strip()
        if part:
            urls.append(part)
    if len(urls) > 16:
        raise ValueError("参考图片URL最多支持 16 张")
    return urls


def _normalize_image_url(api_base, endpoint):
    base = (api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("接口地址不能为空")
    if base.endswith("/images/generations") or base.endswith("/images/edits"):
        return base
    if base.endswith("/v1"):
        return f"{base}/images/{endpoint}"
    return f"{base}/v1/images/{endpoint}"


def _normalize_v1_url(api_base, path):
    base = (api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("接口地址不能为空")
    parsed = urllib.parse.urlsplit(base)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("接口地址必须是完整 URL")
    root = f"{parsed.scheme}://{parsed.netloc}"
    if parsed.path.startswith("/v1"):
        return f"{root}/v1/{path.lstrip('/')}"
    return f"{base}/v1/{path.lstrip('/')}"


def _use_apimart_protocol(api_base, model, selected_protocol):
    if selected_protocol == "APIMart异步":
        return True
    if selected_protocol == "OpenAI同步":
        return False
    base = (api_base or "").lower()
    return "apimart.ai" in base or str(model or "").endswith("-official")


def _read_error_body(exc):
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def _parse_stream_response(resp, completed_type):
    final_event = None
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if event.get("type") == completed_type or event.get("b64_json"):
            final_event = event
    if final_event is None:
        raise RuntimeError("流式响应没有返回最终图片")
    return {"data": [{"b64_json": final_event["b64_json"]}], "stream_event": final_event}


def _request_json(url, api_key, payload, timeout, stream_type=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream_type else "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream_type:
                return _parse_stream_response(resp, stream_type)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"接口请求失败 HTTP {exc.code}: {_read_error_body(exc)}") from exc
    except http.client.RemoteDisconnected as exc:
        raise RuntimeError("接口连接被后台提前断开；请使用流式生成，或稍后重试") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"接口连接失败: {exc}") from exc


def _request_get_json(url, api_key, timeout):
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"任务查询失败 HTTP {exc.code}: {_read_error_body(exc)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"任务查询连接失败: {exc}") from exc


def _multipart_body(fields, files):
    boundary = f"----mumu-gpt-image-2-{uuid.uuid4().hex}"
    chunks = []
    for name, value in fields.items():
        if value is None:
            continue
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, filename, content_type, content in files:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(chunks)


def _request_multipart(url, api_key, fields, files, timeout, stream_type=None):
    boundary, body = _multipart_body(fields, files)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "text/event-stream" if stream_type else "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if stream_type:
                return _parse_stream_response(resp, stream_type)
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"图生图接口请求失败 HTTP {exc.code}: {_read_error_body(exc)}") from exc
    except http.client.RemoteDisconnected as exc:
        raise RuntimeError("图生图接口连接被后台提前断开；节点已建议使用流式生成，或稍后重试") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"图生图接口连接失败: {exc}") from exc


def _upload_image_file(api_base, api_key, file_tuple, timeout):
    _, filename, content_type, content = file_tuple
    endpoint = _normalize_v1_url(api_base, "uploads/images")
    response = _request_multipart(
        endpoint,
        api_key,
        {},
        [("file", filename, content_type, content)],
        timeout,
    )
    image_url = response.get("url") or (response.get("data") or {}).get("url")
    if not image_url:
        raise RuntimeError(f"上传图片没有返回 url: {json.dumps(response, ensure_ascii=False)[:1000]}")
    return image_url


def _extract_task_id(response):
    data = response.get("data")
    candidates = []
    if isinstance(data, list):
        candidates.extend(item for item in data if isinstance(item, dict))
    elif isinstance(data, dict):
        candidates.append(data)
    candidates.append(response)
    for item in candidates:
        task_id = item.get("task_id") or item.get("id")
        status = item.get("status")
        if task_id and status in {None, "submitted", "in_progress", "pending", "queued"}:
            return task_id
    return None


def _poll_task_result(api_base, api_key, task_id, timeout):
    deadline = time.time() + timeout
    last_response = None
    while time.time() < deadline:
        url = _normalize_v1_url(api_base, f"tasks/{urllib.parse.quote(task_id)}?language=zh")
        last_response = _request_get_json(url, api_key, min(30, max(5, int(deadline - time.time()))))
        data = last_response.get("data") if isinstance(last_response.get("data"), dict) else last_response
        status = data.get("status") if isinstance(data, dict) else None
        if status == "completed":
            return last_response
        if status == "failed":
            raise RuntimeError(f"任务失败: {json.dumps(last_response, ensure_ascii=False)[:1000]}")
        time.sleep(3)
    raise RuntimeError(f"等待任务超时，最后响应: {json.dumps(last_response, ensure_ascii=False)[:1000]}")


def _download_image_url(url, timeout):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            image = Image.open(BytesIO(resp.read())).convert("RGB")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下载图片失败: {exc}") from exc
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _collect_result_urls(response):
    urls = []

    def walk(value):
        if isinstance(value, dict):
            if "b64_json" in value:
                return
            url_value = value.get("url")
            if isinstance(url_value, str):
                urls.append(url_value)
            elif isinstance(url_value, list):
                urls.extend(item for item in url_value if isinstance(item, str))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(response)
    return urls


def _image_to_tensor(b64_json):
    image_bytes = base64.b64decode(b64_json)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _resize_tensor_to_size(tensor, size):
    if not size or size == "auto":
        return tensor
    if tensor.ndim == 4:
        resized_batch = [_resize_tensor_to_size(image, size) for image in tensor]
        return torch.stack(resized_batch, dim=0)
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        return tensor
    target_width = int(match.group(1))
    target_height = int(match.group(2))
    array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    source_width, source_height = image.size
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if source_ratio > target_ratio:
        crop_width = int(source_height * target_ratio)
        left = max(0, (source_width - crop_width) // 2)
        image = image.crop((left, 0, left + crop_width, source_height))
    elif source_ratio < target_ratio:
        crop_height = int(source_width / target_ratio)
        top = max(0, (source_height - crop_height) // 2)
        image = image.crop((0, top, source_width, top + crop_height))
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    resized = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(resized)


def _resize_tensors_to_size(tensors, size):
    return [_resize_tensor_to_size(tensor, size) for tensor in tensors]


def _tensor_to_png_files(image_tensor, start_index=0):
    if image_tensor is None:
        return []
    files = []
    tensor = image_tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    for index, image in enumerate(tensor):
        if start_index + index >= 16:
            raise ValueError("参考图片最多支持 16 张")
        array = (image.numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(array[:, :, :3], mode="RGB")
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        files.append(("image[]", f"reference_{start_index + index + 1}.png", "image/png", buf.getvalue()))
    return files


def _collect_reference_image_files(kwargs):
    files = []
    legacy_image = kwargs.get("参考图片")
    if legacy_image is not None:
        files.extend(_tensor_to_png_files(legacy_image, len(files)))
    for key in REFERENCE_IMAGE_KEYS:
        image = kwargs.get(key)
        if image is not None:
            files.extend(_tensor_to_png_files(image, len(files)))
    if len(files) > 16:
        raise ValueError("参考图片最多支持 16 张")
    return files


def _mask_to_png_file(mask_tensor):
    if mask_tensor is None:
        return None
    mask = mask_tensor.detach().cpu().clamp(0, 1)
    if mask.ndim == 3:
        mask = mask[0]
    alpha = ((1.0 - mask.numpy()) * 255).astype(np.uint8)
    rgba = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = alpha
    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return ("mask", "mask.png", "image/png", buf.getvalue())


def _safe_prefix(prefix):
    prefix = (prefix or "mumu_gpt_image_2").strip()
    prefix = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", prefix)
    return prefix or "mumu_gpt_image_2"


def _save_images(tensors, output_format, filename_prefix):
    if folder_paths is None:
        output_dir = os.path.join(NODE_DIR, "output")
    else:
        output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    fmt = (output_format or "png").lower()
    if fmt not in {"png", "jpeg", "webp"}:
        fmt = "png"
    suffix = "jpg" if fmt == "jpeg" else fmt
    saved = []
    prefix = _safe_prefix(filename_prefix)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i, tensor in enumerate(tensors):
        array = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        image = Image.fromarray(array)
        path = os.path.join(output_dir, f"{prefix}_{stamp}_{i + 1:02d}.{suffix}")
        image.save(path, format=fmt.upper() if fmt != "jpeg" else "JPEG")
        saved.append(path)
    return "\n".join(saved)


def _base_payload(kwargs):
    size = _resolve_size(
        kwargs.get("画面比例", kwargs.get("尺寸")),
        kwargs.get("清晰度", kwargs.get("分辨率档位")),
        kwargs.get("自定义尺寸"),
    )
    payload = {
        "model": kwargs.get("模型", "gpt-image-2").strip() or "gpt-image-2",
        "prompt": (kwargs.get("提示词：") or kwargs.get("提示词") or "").strip(),
        "n": int(kwargs.get("图片数量", 1)),
    }
    optional_map = {
        "background": kwargs.get("背景"),
        "moderation": kwargs.get("审核强度"),
        "output_format": kwargs.get("输出格式"),
        "quality": kwargs.get("质量"),
        "user": kwargs.get("用户标识"),
    }
    for key, value in optional_map.items():
        clean_value = _clean_optional(value)
        if clean_value is not None:
            payload[key] = clean_value
    if size is not None:
        payload["size"] = size
    if payload.get("output_format") in {"jpeg", "webp"}:
        payload["output_compression"] = int(kwargs.get("输出压缩", 100))
    stream = bool(kwargs.get("流式生成", False))
    partial_images = int(kwargs.get("局部预览数量", 0))
    if stream:
        payload["stream"] = True
        payload["partial_images"] = partial_images
    elif partial_images > 0:
        payload["partial_images"] = partial_images
    return payload


class MumuGPTImage2:
    @classmethod
    def INPUT_TYPES(cls):
        config = _load_config()
        return {
            "required": {
                "模式": (["自动", "文生图", "图生图"], {"default": "自动"}),
                "提示词：": ("STRING", {"multiline": True, "default": "一张清爽的产品海报，白色背景，柔和影棚光，中文标题：木木 AI"}),
                "接口地址": ("STRING", {"default": config["api_base"]}),
                "key：": ("STRING", {"default": config["api_key"]}),
                "模型": ("STRING", {"default": "gpt-image-2"}),
                "画面比例": (ASPECT_RATIO_OPTIONS, {"default": "1:1"}),
                "质量": (["auto", "low", "medium", "high", "standard", "hd"], {"default": "auto"}),
                "输出格式": (["png", "jpeg", "webp", OMIT], {"default": "png"}),
                "图片数量": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "清晰度": (RESOLUTION_OPTIONS, {"default": "1k"}),
                **{key: ("IMAGE",) for key in REFERENCE_IMAGE_KEYS},
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = "木木节点"

    def generate(self, **kwargs):
        prompt = (kwargs.get("提示词：") or kwargs.get("提示词") or "").strip()
        api_key = (kwargs.get("key：") or kwargs.get("key") or kwargs.get("API密钥") or "").strip()
        api_base = kwargs.get("接口地址", "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        if not api_key:
            raise ValueError("key不能为空")

        reference_files = _collect_reference_image_files(kwargs)
        reference_urls = _split_image_urls(kwargs.get("参考图片URL"))
        mode = kwargs.get("模式", "自动")
        use_edit = mode == "图生图" or (mode == "自动" and (reference_files or reference_urls))

        payload = _base_payload(kwargs)
        timeout = int(kwargs.get("超时时间秒", 180))
        stream_type = None
        apimart_mode = _use_apimart_protocol(api_base, payload.get("model"), kwargs.get("请求协议", "自动"))

        if apimart_mode:
            if use_edit and not reference_files and not reference_urls:
                raise ValueError("图生图模式需要连接参考图片或填写参考图片URL")
            if reference_files:
                uploaded_urls = [_upload_image_file(api_base, api_key, file_tuple, timeout) for file_tuple in reference_files]
                reference_urls = [*reference_urls, *uploaded_urls]
            if reference_urls:
                payload["image_urls"] = reference_urls
                mask_file = _mask_to_png_file(kwargs.get("遮罩"))
                if mask_file is not None:
                    payload["mask_url"] = _upload_image_file(api_base, api_key, mask_file, timeout)
                mask_url = (kwargs.get("遮罩URL") or "").strip()
                if mask_url:
                    payload["mask_url"] = mask_url
            endpoint = _normalize_image_url(api_base, "generations")
            response = _request_json(endpoint, api_key, payload, timeout)
            task_id = _extract_task_id(response)
            if task_id:
                task_response = _poll_task_result(api_base, api_key, task_id, timeout)
                response = {"提交响应": response, "任务结果": task_response}
        elif use_edit:
            if not reference_files and not reference_urls:
                raise ValueError("图生图模式需要连接参考图片或填写参考图片URL")
            # aimumu's GPT-Image-2 edit route is backed by an internal image tool
            # that rejects n as tools[0].n, so edit requests always ask for one image.
            payload.pop("n", None)
            # The same route can close long non-streaming edit requests, so keep edits streaming.
            payload["stream"] = True
            payload.setdefault("partial_images", 0)
            input_fidelity = _clean_optional(kwargs.get("输入保真度"))
            if input_fidelity is not None:
                payload["input_fidelity"] = input_fidelity
            endpoint = _normalize_image_url(api_base, "edits")
            if reference_files:
                files = reference_files
                mask_file = _mask_to_png_file(kwargs.get("遮罩"))
                if mask_file is not None:
                    files.append(mask_file)
                fields = {key: ("true" if value is True else "false" if value is False else value) for key, value in payload.items()}
                if bool(payload.get("stream", False)):
                    stream_type = "image_edit.completed"
                response = _request_multipart(endpoint, api_key, fields, files, timeout, stream_type)
            else:
                payload["images"] = [{"image_url": url} for url in reference_urls]
                mask_url = (kwargs.get("遮罩URL") or "").strip()
                if mask_url:
                    payload["mask"] = {"image_url": mask_url}
                if bool(payload.get("stream", False)):
                    stream_type = "image_edit.completed"
                response = _request_json(endpoint, api_key, payload, timeout, stream_type)
        else:
            response_format = _clean_optional(kwargs.get("响应格式"))
            if response_format is not None:
                payload["response_format"] = response_format
            style = _clean_optional(kwargs.get("风格"))
            if style is not None:
                payload["style"] = style
            endpoint = _normalize_image_url(api_base, "generations")
            if bool(payload.get("stream", False)):
                stream_type = "image_generation.completed"
            response = _request_json(endpoint, api_key, payload, timeout, stream_type)

        data = response.get("data") or []
        tensors = []
        for item in data:
            b64_json = item.get("b64_json")
            if b64_json:
                tensors.append(_image_to_tensor(b64_json))
        if not tensors:
            for image_url in _collect_result_urls(response):
                tensors.append(_download_image_url(image_url, timeout))
        if not tensors:
            raise RuntimeError(f"接口没有返回 b64_json 图片: {json.dumps(response, ensure_ascii=False)[:1000]}")
        if bool(kwargs.get("输出匹配尺寸", True)):
            tensors = _resize_tensors_to_size(tensors, payload.get("size"))

        images = torch.cat(tensors, dim=0)
        if bool(kwargs.get("保存到输出目录", True)):
            _save_images(images, payload.get("output_format"), kwargs.get("文件名前缀", "mumu_gpt_image_2"))

        return (images,)


NODE_CLASS_MAPPINGS = {
    "MumuGPTImage2": MumuGPTImage2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MumuGPTImage2": "玩AI的木木--GPT Image2 图像生成",
}
