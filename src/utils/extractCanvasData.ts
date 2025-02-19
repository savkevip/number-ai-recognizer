import { IMAGE_SIZE, IMAGE_SIZES } from "./constants";

export function getCanvasData(canvas: HTMLCanvasElement): number[] {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context not available");

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = IMAGE_SIZE;
  tempCanvas.height = IMAGE_SIZE;
  const tempCtx = tempCanvas.getContext("2d");
  if (!tempCtx) throw new Error("Temporary canvas context not available");

  const boundingBox = getBoundingBox(canvas);
  if (!boundingBox) return new Array(IMAGE_SIZES).fill(0);

  const scale = Math.min(
    IMAGE_SIZE / boundingBox.width,
    IMAGE_SIZE / boundingBox.height
  );

  const scaledWidth = boundingBox.width * scale;
  const scaledHeight = boundingBox.height * scale;

  const offsetX = (IMAGE_SIZE - scaledWidth) / 2;
  const offsetY = (IMAGE_SIZE - scaledHeight) / 2;

  tempCtx.drawImage(
    canvas,
    boundingBox.x,
    boundingBox.y,
    boundingBox.width,
    boundingBox.height,
    offsetX,
    offsetY,
    scaledWidth,
    scaledHeight
  );

  const imageData = tempCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const pixels = imageData.data;

  const normalizedPixels: number[] = [];

  for (let i = 0; i < pixels.length; i += 4) {
    const grayscale = pixels[i];
    const normalized = grayscale / 255;
    normalizedPixels.push(normalized);
  }

  console.log("Max Pixel Value:", Math.max(...normalizedPixels));
  console.log("Min Pixel Value:", Math.min(...normalizedPixels));

  return normalizedPixels;
}

function getBoundingBox(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = imageData.data;

  let minX = canvas.width,
    minY = canvas.height,
    maxX = 0,
    maxY = 0;
  let found = false;

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      const grayscale = pixels[i];

      if (grayscale < 128) {
        found = true;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!found) return null;

  const padding = 4;
  return {
    x: Math.max(0, minX - padding),
    y: Math.max(0, minY - padding),
    width: Math.min(canvas.width, maxX - minX + 2 * padding),
    height: Math.min(canvas.height, maxY - minY + 2 * padding),
  };
}
