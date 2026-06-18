import { app } from "../../scripts/app.js";

const NODE_TYPE = "MumuGPTImage2";
const REF_PREFIX = "\u53c2\u8003\u56fe\u7247";
const MAX_REFS = 16;
const INPUT = 1;
const HIDDEN_WIDGET_NAMES = new Set(["\u6a21\u578b", "\u8d28\u91cf"]);
const HIDDEN_OUTPUT_NAMES = new Set(["\u63a5\u53e3\u54cd\u5e94JSON", "\u4fdd\u5b58\u8def\u5f84"]);
const ASPECT_RATIOS = new Set(["auto", "1:1", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "16:9", "9:16", "2:1", "1:2", "3:1", "1:3", "21:9", "9:21"]);
const RESOLUTIONS = new Set(["1k", "2k", "4k", "\u4e0d\u53d1\u9001"]);
const OUTPUT_FORMATS = new Set(["png", "jpeg", "webp", "\u4e0d\u53d1\u9001"]);
const QUALITIES = new Set(["auto", "low", "medium", "high", "standard", "hd"]);

function refNumber(name) {
  const text = String(name || "");
  if (!text.startsWith(REF_PREFIX)) {
    return 0;
  }
  const value = Number(text.slice(REF_PREFIX.length));
  return Number.isInteger(value) ? value : 0;
}

function refInputs(node) {
  return (node.inputs || []).filter((input) => refNumber(input.name) > 0);
}

function hasLink(input) {
  return input?.link != null || (Array.isArray(input?.links) && input.links.length > 0);
}

function markDirty(node) {
  node.setSize?.(node.computeSize?.() || node.size);
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
}

function hideAdvancedWidgets(node) {
  let changed = false;
  for (const widget of node?.widgets || []) {
    if (!HIDDEN_WIDGET_NAMES.has(widget?.name) || widget._mumuHidden) {
      continue;
    }
    widget._mumuOriginalType = widget.type;
    widget.computeSize = () => [0, -4];
    widget.serialize = true;
    widget._mumuHidden = true;
    changed = true;
  }
  if (changed) {
    markDirty(node);
  }
}

function widgetByName(node, name) {
  return (node?.widgets || []).find((widget) => widget?.name === name);
}

function setWidgetValue(widget, value) {
  if (widget && value !== undefined && value !== null) {
    widget.value = value;
  }
}

function validInt(value, min, max) {
  const number = Number(value);
  if (!Number.isInteger(number)) {
    return null;
  }
  if (number < min || number > max) {
    return null;
  }
  return number;
}

function repairShiftedWidgetValues(node) {
  const model = widgetByName(node, "\u6a21\u578b");
  const ratio = widgetByName(node, "\u753b\u9762\u6bd4\u4f8b");
  const quality = widgetByName(node, "\u8d28\u91cf");
  const outputFormat = widgetByName(node, "\u8f93\u51fa\u683c\u5f0f");
  const imageCount = widgetByName(node, "\u56fe\u7247\u6570\u91cf");
  const resolution = widgetByName(node, "\u6e05\u6670\u5ea6");
  const values = node?.widgets_values || [];
  let changed = false;

  if (ratio && !ASPECT_RATIOS.has(String(ratio.value))) {
    if (model && String(ratio.value || "").startsWith("gpt-image")) {
      setWidgetValue(model, ratio.value);
      changed = true;
    }
    if (outputFormat && ASPECT_RATIOS.has(String(outputFormat.value))) {
      setWidgetValue(ratio, outputFormat.value);
      changed = true;
    } else {
      setWidgetValue(ratio, "1:1");
      changed = true;
    }
    if (quality && imageCount && QUALITIES.has(String(imageCount.value))) {
      setWidgetValue(quality, imageCount.value);
      changed = true;
    }
    if (outputFormat && resolution && OUTPUT_FORMATS.has(String(resolution.value))) {
      setWidgetValue(outputFormat, resolution.value);
      changed = true;
    }
    const oldCount = validInt(values[8], 1, 4);
    if (imageCount) {
      setWidgetValue(imageCount, oldCount ?? 1);
      changed = true;
    }
    if (resolution) {
      const oldResolution = RESOLUTIONS.has(String(values[9])) ? values[9] : "1k";
      setWidgetValue(resolution, oldResolution);
      changed = true;
    }
  }

  if (ratio && !ASPECT_RATIOS.has(String(ratio.value))) {
    setWidgetValue(ratio, "1:1");
    changed = true;
  }
  if (quality && !QUALITIES.has(String(quality.value))) {
    setWidgetValue(quality, "auto");
    changed = true;
  }
  if (outputFormat && !OUTPUT_FORMATS.has(String(outputFormat.value))) {
    setWidgetValue(outputFormat, "png");
    changed = true;
  }
  if (imageCount) {
    const count = validInt(imageCount.value, 1, 4);
    if (count === null) {
      setWidgetValue(imageCount, 1);
      changed = true;
    } else if (imageCount.value !== count) {
      setWidgetValue(imageCount, count);
      changed = true;
    }
  }
  if (resolution && !RESOLUTIONS.has(String(resolution.value))) {
    setWidgetValue(resolution, "1k");
    changed = true;
  }

  if (changed) {
    markDirty(node);
  }
}

function removeLegacyOutputs(node) {
  let changed = false;
  for (let index = (node?.outputs || []).length - 1; index >= 0; index -= 1) {
    const output = node.outputs[index];
    if (HIDDEN_OUTPUT_NAMES.has(output?.name)) {
      node.removeOutput(index);
      changed = true;
    }
  }
  if (changed) {
    markDirty(node);
  }
}

function isMumuNode(node) {
  return (
    node?.comfyClass === NODE_TYPE ||
    node?.type === NODE_TYPE ||
    node?.constructor?.comfyClass === NODE_TYPE ||
    node?.constructor?.nodeData?.name === NODE_TYPE
  );
}

function ensureAllMumuNodes() {
  const graph = app.graph || app.canvas?.graph || app.canvas?.getCurrentGraph?.();
  for (const node of graph?._nodes || []) {
    if (isMumuNode(node)) {
      prepareMumuNode(node);
    }
  }
}

function scheduleEnsure(node) {
  for (const delay of [0, 100, 500, 1000, 2000, 4000]) {
    setTimeout(() => prepareMumuNode(node), delay);
  }
}

function ensureDynamicReferenceInputs(node) {
  if (!node?.inputs) {
    return;
  }

  const refs = refInputs(node);
  const connectedIndexes = refs
    .filter(hasLink)
    .map((input) => refNumber(input.name));
  const maxConnected = connectedIndexes.length ? Math.max(...connectedIndexes) : 0;
  const visibleCount = Math.min(MAX_REFS, Math.max(1, maxConnected + 1));

  for (let index = 1; index <= visibleCount; index += 1) {
    const name = `${REF_PREFIX}${index}`;
    if (!node.inputs.some((input) => input.name === name)) {
      node.addInput(name, "IMAGE");
    }
  }

  for (let index = node.inputs.length - 1; index >= 0; index -= 1) {
    const input = node.inputs[index];
    const refIndex = refNumber(input?.name);
    if (refIndex > visibleCount && refIndex > 0 && !hasLink(input)) {
      node.removeInput(index);
    }
  }

  markDirty(node);
}

function shouldHandleConnection(node, type, input) {
  return isMumuNode(node) && (type === undefined || type === INPUT) && refNumber(input?.name) > 0;
}

function prepareMumuNode(node) {
  repairShiftedWidgetValues(node);
  hideAdvancedWidgets(node);
  removeLegacyOutputs(node);
  ensureDynamicReferenceInputs(node);
}

app.registerExtension({
  name: "mumu.gpt-image-2.dynamic-reference-inputs",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      scheduleEnsure(this);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      scheduleEnsure(this);
      return result;
    };

    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, slotIndex, connected, linkInfo, ioSlot) {
      const result = originalOnConnectionsChange?.apply(this, arguments);
      const input = ioSlot || this.inputs?.[slotIndex];
      if (shouldHandleConnection(this, type, input) || linkInfo) {
        scheduleEnsure(this);
      }
      return result;
    };
  },
  nodeCreated(node) {
    if (isMumuNode(node)) {
      scheduleEnsure(node);
    }
  },
  loadedGraphNode(node) {
    if (isMumuNode(node)) {
      scheduleEnsure(node);
    }
  },
  setup() {
    let attempts = 0;
    const timer = setInterval(() => {
      attempts += 1;
      ensureAllMumuNodes();
      if (attempts >= 20) {
        clearInterval(timer);
      }
    }, 500);
  },
});
