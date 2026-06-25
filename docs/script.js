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
    id: "omni-02",
    type: "Omnidirectional",
    name: "Pillar",
    fov: "Ricoh360 · 360° × 180°",
    icon: "assets/Exp1_Result/Omni_2/icon.png",
    methods: [
      { label: "GT", src: "assets/Exp1_Result/Omni_2/gt.png" },
      { label: "ODGS", src: "assets/Exp1_Result/Omni_2/ODGS.png" },
      { label: "OP43DGS", src: "assets/Exp1_Result/Omni_2/OP43DGS.png" },
      { label: "OmniGS", src: "assets/Exp1_Result/Omni_2/OmniGS.png" },
      { label: "Ours", src: "assets/Exp1_Result/Omni_2/Ours.png" },
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
    type: "Fisheye",
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
    type: "Fisheye",
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
    if (dicsInstances.has(index)) return;
    const container = scenesHost.querySelector(`.b-dics${index}`);
    if (!container || typeof Dics === "undefined") return;

    dicsInstances.set(index, new Dics({
      container,
      textPosition: "top",
    }));
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

  tabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-scene-index]");
    if (!tab) return;
    showScene(Number(tab.dataset.sceneIndex));
  });

  showScene(0);
}

const dialog = document.querySelector(".image-dialog");
const dialogImage = dialog?.querySelector("img");
const closeButton = dialog?.querySelector(".dialog-close");

document.querySelectorAll(".zoom-trigger").forEach((trigger) => {
  trigger.addEventListener("click", () => {
    if (!dialog || !dialogImage) return;
    dialogImage.src = trigger.dataset.full;
    dialogImage.alt = trigger.querySelector("img")?.alt || "Expanded research figure";
    dialog.showModal();
  });
});

closeButton?.addEventListener("click", () => dialog?.close());

dialog?.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});
