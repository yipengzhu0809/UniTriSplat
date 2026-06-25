const heroVideo = document.querySelector(".hero-video");

if (heroVideo) {
  heroVideo.playbackRate = 2;
}

const copyButton = document.querySelector("[data-copy-target]");

copyButton?.addEventListener("click", async () => {
  const target = document.getElementById(copyButton.dataset.copyTarget);
  if (!target) return;

  try {
    await navigator.clipboard.writeText(target.innerText);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1800);
  } catch {
    copyButton.textContent = "Select and copy";
  }
});


const multiFovScenes = [
  {
    id: "omni-01",
    type: "Omnidirectional",
    name: "Office",
    fov: "360Roam · 360° × 180°",
    icon: "assets/Exp1_Result/Omni_1/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Omni_1/GT.png" },
      { label: "ODGS", src: "assets/Exp1_Result/Omni_1/ODGS.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Omni_1/OP43DGS.png" },
      { label: "OmniGS", src: "assets/Exp1_Result/Omni_1/OmniGS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Omni_1/ours.png" },
    ],
  },
  {
    id: "omni-03",
    type: "Omnidirectional",
    name: "Chair",
    fov: "OmniBlender · 360° × 180°",
    icon: "assets/Exp1_Result/Omni_3/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Omni_3/gt.png" },
      { label: "ODGS", src: "assets/Exp1_Result/Omni_3/ODGS.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Omni_3/OP43DGS.png" },
      { label: "OmniGS", src: "assets/Exp1_Result/Omni_3/OmniGS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Omni_3/Ours.png" },
    ],
  },
  {
    id: "fisheye-01",
    type: "Ultra-wide Fisheye",
    name: "Bridge",
    fov: "FIORD · 155° × 155°",
    icon: "assets/Exp1_Result/Fisheye_1/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Fisheye_1/gt.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Fisheye_1/OP43DGS.png" },
      { label: "Fisheye-GS", src: "assets/Exp1_Result/Fisheye_1/Fisheye-GS.png" },
      { label: "3DGUT", src: "assets/Exp1_Result/Fisheye_1/3DGUT.png" },
      { label: "Ours", src: "assets/Exp1_Result/Fisheye_1/Ours.png" },
    ],
  },
  {
    id: "fisheye-02",
    type: "Ultra-wide Fisheye",
    name: "Meetingroom",
    fov: "FIORD · 135° × 135°",
    icon: "assets/Exp1_Result/Fisheye_2/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Fisheye_2/gt.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Fisheye_2/OP43DGS.png" },
      { label: "Fisheye-GS", src: "assets/Exp1_Result/Fisheye_2/Fisheye-GS.png" },
      { label: "3DGUT", src: "assets/Exp1_Result/Fisheye_2/3DGUT.png" },
      { label: "Ours", src: "assets/Exp1_Result/Fisheye_2/Ours.png" },
    ],
  },
  {
    id: "fisheye-03",
    type: "Fisheye",
    name: "Dormitory",
    fov: "ScanNet++ · 127° × 85°",
    icon: "assets/Exp1_Result/Fisheye_3/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Fisheye_3/gt.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Fisheye_3/OP43DGS.png" },
      { label: "Fisheye-GS", src: "assets/Exp1_Result/Fisheye_3/Fisheye-GS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Fisheye_3/Ours.png" },
    ],
  },
  {
    id: "fisheye-04",
    type: "Fisheye",
    name: "Lab",
    fov: "ScanNet++ · 127° × 85°",
    icon: "assets/Exp1_Result/Fisheye_4/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Fisheye_4/gt.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Fisheye_4/OP43DGS.png" },
      { label: "Fisheye-GS", src: "assets/Exp1_Result/Fisheye_4/Fisheye-GS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Fisheye_4/Ours.png" },
    ],
  },
  {
    id: "perspective-01",
    type: "Perspective",
    name: "Flower",
    fov: "Mip-NeRF 360 · 60° × 45°",
    icon: "assets/Exp1_Result/Pp_1/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Pp_1/gt.png" },
      { label: "3DGS", src: "assets/Exp1_Result/Pp_1/3DGS.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Pp_1/OP43DGS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Pp_1/Ours.png" },
    ],
  },
  {
    id: "perspective-02",
    type: "Perspective",
    name: "Treehill",
    fov: "Mip-NeRF 360 · 62° × 43°",
    icon: "assets/Exp1_Result/Pp_2/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Pp_2/gt.png" },
      { label: "3DGS", src: "assets/Exp1_Result/Pp_2/3DGS.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Pp_2/OP43DGS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Pp_2/Ours.png" },
    ],
  },
];

const multiFovViewer = document.querySelector("[data-multifov-viewer]");

if (multiFovViewer) {
  const scenesHost = multiFovViewer.querySelector("[data-dics-scenes]");
  const tabs = multiFovViewer.querySelector("[data-scene-tabs]");
  const sceneType = multiFovViewer.querySelector("[data-scene-type]");
  const sceneName = multiFovViewer.querySelector("[data-scene-name]");
  const sceneFov = multiFovViewer.querySelector("[data-scene-fov]");
  const dicsInstances = new Map();
  let activeScene = 0;

  const sceneHasImages = (scene) => scene.methods.filter((method) => method.src).length >= 2;

  const waitForImage = (image) => {
    if (!image) return Promise.resolve(null);
    if (image.complete && image.naturalWidth) return Promise.resolve(image);

    return new Promise((resolve) => {
      image.addEventListener("load", () => resolve(image), { once: true });
      image.addEventListener("error", () => resolve(null), { once: true });
    });
  };

  const fitDicsContainerToFrame = (container) => {
    const firstImage = container?.querySelector("img");
    if (!container || !firstImage?.naturalWidth || !firstImage?.naturalHeight) return;

    const frame = scenesHost.getBoundingClientRect();
    const imageRatio = firstImage.naturalWidth / firstImage.naturalHeight;
    const fittedWidth = Math.min(frame.width, frame.height * imageRatio);
    container.style.width = `${Math.max(1, Math.floor(fittedWidth))}px`;
    container.style.maxWidth = "100%";
  };

  const refreshDics = (index) => {
    const instance = dicsInstances.get(index);
    if (!instance || instance === "pending") return;

    fitDicsContainerToFrame(instance.container);
    instance._setContainerWidth(instance.images[0]);
    instance._resetSizes();
  };

  scenesHost.innerHTML = multiFovScenes.map((scene, index) => {
    const images = scene.methods
      .filter((method) => method.src)
      .map((method) => `<img src="${method.src}" alt="${method.label}" draggable="false">`)
      .join("");

    if (!images) {
      return `<div class="sc-dics-scene sc-dics-empty is-hidden" id="compare-res${index}">
        <p>Add images for ${scene.name}</p>
      </div>`;
    }

    return `<div class="sc-dics-scene is-hidden" id="compare-res${index}">
      <div class="b-dics b-dics${index}" style="width: 100%">${images}</div>
    </div>`;
  }).join("");

  tabs.innerHTML = multiFovScenes.map((scene, index) => {
    const icon = scene.icon
      ? `<img src="${scene.icon}" alt="" loading="lazy">`
      : `<span>${index + 1}</span>`;
    return `<button class="fov-tab carousel-btn" type="button" role="tab" aria-selected="${index === 0}" id="compare-btn-${index}" data-scene-index="${index}">
      <span class="fov-icon">${icon}</span>
      <strong>${scene.type}</strong>
      <span>${scene.fov.split(" · ").pop()}</span>
    </button>`;
  }).join("");

  const initializeDics = (index) => {
    if (dicsInstances.has(index)) {
      refreshDics(index);
      return;
    }

    const container = scenesHost.querySelector(`.b-dics${index}`);
    if (!container || typeof Dics === "undefined") return;

    dicsInstances.set(index, "pending");
    waitForImage(container.querySelector("img")).then((image) => {
      if (!image) {
        dicsInstances.delete(index);
        return;
      }

      fitDicsContainerToFrame(container);
      dicsInstances.set(index, new Dics({
        container,
        textPosition: "top",
      }));
    });
  };

  const showScene = (index) => {
    const scene = multiFovScenes[index];
    if (!scene) return;

    activeScene = index;
    sceneType.textContent = scene.type;
    sceneName.textContent = scene.name;
    sceneFov.textContent = scene.fov;

    scenesHost.querySelectorAll(".sc-dics-scene").forEach((element, sceneIndex) => {
      element.classList.toggle("is-hidden", sceneIndex !== activeScene);
    });

    tabs.querySelectorAll(".fov-tab").forEach((tab, tabIndex) => {
      tab.classList.toggle("is-active", tabIndex === activeScene);
      tab.classList.toggle("active", tabIndex === activeScene);
      tab.setAttribute("aria-selected", String(tabIndex === activeScene));
    });

    if (sceneHasImages(scene)) {
      requestAnimationFrame(() => initializeDics(index));
    }
  };

  window.addEventListener("resize", () => {
    requestAnimationFrame(() => refreshDics(activeScene));
  });

  tabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-scene-index]");
    if (!tab) return;
    showScene(Number(tab.dataset.sceneIndex));
  });

  showScene(0);
}


const crossCameraScenes = [
  {
    id: "pillar",
    name: "Pillar",
    dataset: "Ricoh360",
    folder: "Pillar",
    note: "Reconstruction quality of each method on the omnidirectional scene",
    fovs: { fisheye: "120°", perspective: "45°" },
  },
  {
    id: "center",
    name: "Center",
    dataset: "360Roam",
    folder: "Center",
    note: "Reconstruction quality of each method on the omnidirectional scene",
    fovs: { fisheye: "240°", perspective: "90°" },
  },
];

const crossCameraModes = {
  fisheye: {
    label: "Fisheye",
    note: "Render with fisheye-specific rasterizers vs. our unified rasterizer",
    renderMethods: ["OP43DGS", "Ours"],
  },
  perspective: {
    label: "Perspective",
    note: "Render with perspective-specific rasterizers vs. our unified rasterizer",
    renderMethods: ["3DGS", "Ours"],
  },
};

const crossOptimizedMethods = ["ODGS", "OmniGS", "OP43DGS", "Ours"];
const crossModelMethods = ["gt", "ODGS", "OmniGS", "OP43DGS", "Ours"];
const crossCameraViewer = document.querySelector("[data-cross-camera-viewer]");

if (crossCameraViewer) {
  const sceneTabs = crossCameraViewer.querySelector("[data-cross-scene-tabs]");
  const modelScenesHost = crossCameraViewer.querySelector("[data-cross-model-scenes]");
  const renderTabs = crossCameraViewer.querySelector("[data-cross-render-tabs]");
  const renderLayout = crossCameraViewer.querySelector("[data-cross-render-layout]");
  const sceneName = crossCameraViewer.querySelector("[data-cross-scene-name]");
  const sceneDataset = crossCameraViewer.querySelector("[data-cross-dataset]");
  const sceneNote = crossCameraViewer.querySelector("[data-cross-scene-note]");
  const dicsInstances = new Map();
  let activeScene = 0;
  let activeMode = "fisheye";

  const waitForCrossImage = (image) => {
    if (!image) return Promise.resolve(null);
    if (image.complete && image.naturalWidth) return Promise.resolve(image);

    return new Promise((resolve) => {
      image.addEventListener("load", () => resolve(image), { once: true });
      image.addEventListener("error", () => resolve(null), { once: true });
    });
  };

  const fitCrossDicsContainer = (container) => {
    const firstImage = container?.querySelector("img");
    if (!container || !firstImage?.naturalWidth || !firstImage?.naturalHeight) return;

    const frame = modelScenesHost.getBoundingClientRect();
    const imageRatio = firstImage.naturalWidth / firstImage.naturalHeight;
    const fittedWidth = Math.min(frame.width, frame.height * imageRatio);
    container.style.width = `${Math.max(1, Math.floor(fittedWidth))}px`;
    container.style.maxWidth = "100%";
  };

  const refreshCrossDics = (index) => {
    const instance = dicsInstances.get(index);
    if (!instance || instance === "pending") return;

    fitCrossDicsContainer(instance.container);
    instance._setContainerWidth(instance.images[0]);
    instance._resetSizes();
  };

  const getCrossImagePath = (scene, mode, optimizedMethod, renderMethod) => {
    if (scene.id === "center" && mode === "fisheye" && renderMethod === "OP43DGS") {
      return null;
    }

    const optimized = scene.id === "center" && mode === "fisheye" && optimizedMethod === "ODGS"
      ? "ODGGS"
      : optimizedMethod;

    return `assets/Exp2_Result/${scene.folder}/cross/${mode}/${optimized}_${renderMethod}.png`;
  };

  const buildModelDics = () => {
    modelScenesHost.innerHTML = crossCameraScenes.map((scene, index) => {
      const images = crossModelMethods.map((method) => {
        const label = method === "gt" ? "GT" : method;
        return `<img src="assets/Exp2_Result/${scene.folder}/model/${method}.png" alt="${label}" draggable="false">`;
      }).join("");

      return `<div class="cross-model-scene is-hidden" id="cross-model-${index}">
        <div class="b-dics cross-dics${index}" style="width: 100%">${images}</div>
      </div>`;
    }).join("");
  };

  const buildSceneTabs = () => {
    sceneTabs.innerHTML = crossCameraScenes.map((scene, index) => `<button class="cross-scene-tab" type="button" aria-selected="${index === 0}" data-cross-scene-index="${index}">
      <strong>${scene.name}</strong>
      <span>${scene.dataset}</span>
    </button>`).join("");
  };

  const buildRenderTabs = () => {
    renderTabs.innerHTML = Object.entries(crossCameraModes).map(([mode, config]) => `<button class="cross-render-tab" type="button" aria-selected="${mode === activeMode}" data-cross-mode="${mode}">
      <strong>${config.label}</strong>
      <span>${config.note}</span>
    </button>`).join("");
  };

  const initializeCrossDics = (index) => {
    if (dicsInstances.has(index)) {
      refreshCrossDics(index);
      return;
    }

    const container = modelScenesHost.querySelector(`.cross-dics${index}`);
    if (!container || typeof Dics === "undefined") return;

    dicsInstances.set(index, "pending");
    waitForCrossImage(container.querySelector("img")).then((image) => {
      if (!image) {
        dicsInstances.delete(index);
        return;
      }

      fitCrossDicsContainer(container);
      dicsInstances.set(index, new Dics({
        container,
        textPosition: "top",
      }));
    });
  };

  const renderCrossMatrix = () => {
    const scene = crossCameraScenes[activeScene];
    const modeConfig = crossCameraModes[activeMode];
    const gtSrc = `assets/Exp2_Result/${scene.folder}/cross/${activeMode}/gt.png`;

    const headerCells = crossOptimizedMethods.map((method) => `<div class="cross-col-label">${method}</div>`).join("");
    const bodyCells = modeConfig.renderMethods.map((renderMethod) => {
      const row = [`<div class="cross-row-label">${renderMethod}</div>`];
      crossOptimizedMethods.forEach((optimizedMethod) => {
        const src = getCrossImagePath(scene, activeMode, optimizedMethod, renderMethod);
        const missing = scene.id === "center" && activeMode === "fisheye" && renderMethod === "OP43DGS";
        row.push(`<div class="cross-cell">${src
          ? `<div class="cross-cell-button">
              <img src="${src}" alt="${optimizedMethod} optimized model rendered by ${renderMethod}" loading="lazy">
            </div>`
          : `<div class="cross-cell-empty">${missing ? "FoV not available for OP43DGS" : "Not available"}</div>`
        }</div>`);
      });
      return row.join("");
    }).join("");

    renderLayout.innerHTML = `<aside class="cross-gt-card">
      <div class="cross-gt-inner">
        <h4>Ground Truth</h4>
        <div class="cross-gt-button">
          <img src="${gtSrc}" alt="${scene.name} ${modeConfig.label} ground truth" loading="lazy">
        </div>
        <p class="cross-gt-fov">FoV: ${scene.fovs[activeMode]}</p>
      </div>
    </aside>
    <div class="cross-matrix-card">
      <h4>Columns indicate the reconstruction method, and rows indicate the rendering rasterizer.</h4>
      <div class="cross-matrix-scroll">
        <div class="cross-matrix">
          <div class="cross-axis"><span class="cross-axis-render">Render</span><span class="cross-axis-recon">Model</span></div>
          ${headerCells}
          ${bodyCells}
        </div>
      </div>
    </div>`;
  };

  const showCrossScene = (index) => {
    const scene = crossCameraScenes[index];
    if (!scene) return;

    activeScene = index;
    sceneName.textContent = scene.name;
    sceneDataset.textContent = scene.dataset;
    sceneNote.textContent = scene.note;

    modelScenesHost.querySelectorAll(".cross-model-scene").forEach((element, sceneIndex) => {
      element.classList.toggle("is-hidden", sceneIndex !== activeScene);
    });

    sceneTabs.querySelectorAll(".cross-scene-tab").forEach((tab, tabIndex) => {
      tab.classList.toggle("is-active", tabIndex === activeScene);
      tab.setAttribute("aria-selected", String(tabIndex === activeScene));
    });

    requestAnimationFrame(() => initializeCrossDics(index));
    renderCrossMatrix();
  };

  const showCrossMode = (mode) => {
    if (!crossCameraModes[mode]) return;
    activeMode = mode;

    renderTabs.querySelectorAll(".cross-render-tab").forEach((tab) => {
      const isActive = tab.dataset.crossMode === activeMode;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", String(isActive));
    });

    renderCrossMatrix();
  };

  buildModelDics();
  buildSceneTabs();
  buildRenderTabs();

  sceneTabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-cross-scene-index]");
    if (!tab) return;
    showCrossScene(Number(tab.dataset.crossSceneIndex));
  });

  renderTabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-cross-mode]");
    if (!tab) return;
    showCrossMode(tab.dataset.crossMode);
  });

  window.addEventListener("resize", () => {
    requestAnimationFrame(() => refreshCrossDics(activeScene));
  });

  showCrossScene(0);
  showCrossMode(activeMode);
}

const dialog = document.querySelector(".image-dialog");
const dialogImage = dialog?.querySelector("img");
const closeButton = dialog?.querySelector(".dialog-close");

document.addEventListener("click", (event) => {
  const trigger = event.target.closest(".zoom-trigger");
  if (!trigger || !dialog || !dialogImage) return;

  dialogImage.src = trigger.dataset.full;
  dialogImage.alt = trigger.querySelector("img")?.alt || "Expanded research figure";
  dialog.showModal();
});

closeButton?.addEventListener("click", () => dialog?.close());

dialog?.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});
