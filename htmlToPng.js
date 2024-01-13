const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

async function ensureDirectoryExists(directory) {
  try {
    await fs.access(directory);
  } catch (error) {
    await fs.mkdir(directory);
  }
}

async function takeScreenshot(htmlFilePath, outputFilePath) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Load HTML content
  const htmlContent = await fs.readFile(htmlFilePath, 'utf-8');
  await page.setContent(htmlContent);

  // Wait for any lazy-loaded content, adjust the timeout as needed
  await page.waitForTimeout(200);

  // Ensure the output directory exists
  await ensureDirectoryExists(path.dirname(outputFilePath));

  // Take a screenshot without setting viewport
  await page.screenshot({ path: outputFilePath });

  await browser.close();
}

async function main() {
  // Iterate over each HTML file
  let htmlFilePath = process.argv[2];
  const outputFilePath = `${htmlFilePath.replace(".png","converted").replace('.html', '.png')}`;

  // Take screenshot
  await takeScreenshot(htmlFilePath, outputFilePath);
  console.log(`Screenshot taken: ${outputFilePath}`);

  // Wait for 200ms before the next screenshot
  await new Promise(resolve => setTimeout(resolve, 200));
}

main();
