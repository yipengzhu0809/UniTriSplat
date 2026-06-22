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
