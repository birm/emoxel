const puppeteer = require('puppeteer');
const fs = require('fs');

const inputDirectory = 'emoji_html';
const outputDirectory = 'emoji_png';

// Create the output directory if it doesn't exist
if (!fs.existsSync(outputDirectory)) {
  fs.mkdirSync(outputDirectory);
}

async function takeScreenshot(htmlFilePath, outputFilePath) {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Set viewport size
  await page.setViewport({ width: 32, height: 32 });

  // Load HTML content
  const htmlContent = fs.readFileSync(htmlFilePath, 'utf-8');
  await page.setContent(htmlContent);

  // Take a screenshot
  await page.screenshot({ path: outputFilePath });

  await browser.close();
}

// Iterate over each HTML file in the input directory with a delay
const htmlFiles = fs.readdirSync(inputDirectory);

async function processFilesWithDelay() {
  for (const file of htmlFiles) {
    const htmlFilePath = `${inputDirectory}/${file}`;
    const file_name_no_extension = file.split('.')[0];
    const outputFilePath = `${outputDirectory}/${file_name_no_extension}.png`;

    // Take screenshot
    await takeScreenshot(htmlFilePath, outputFilePath)
      .then(() => console.log(`Screenshot taken: ${outputFilePath}`))
      .catch(error => console.error(`Error taking screenshot: ${error.message}`));

    // Delay for 0.2 seconds
    await new Promise(resolve => setTimeout(resolve, 200));
  }
}

processFilesWithDelay();
