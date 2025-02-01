import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import datetime
from pathlib import Path
import requests
from tqdm import tqdm
import logging
import subprocess


BASE_URL = "https://dps.psx.com.pk"
DOWNLOADS_URL = f"{BASE_URL}/downloads"
DOWNLOAD_DIR = Path.cwd() / "PSX_Market_Summary_Playwright"

START_DATE = datetime.date(2024, 1, 1)
END_DATE = datetime.date(2024, 6, 1)
REQUEST_DELAY = 1.0


logging.basicConfig(
    filename='psx_scraper_playwright.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def create_directory(path: Path):
    """Creates a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)

def extract_z_file(file_path: Path):
    """
    Extracts a .Z file using the 'uncompress' utility.
    """
    try:
        subprocess.run(['uncompress', str(file_path)], check=True)
        logging.info(f"Successfully extracted {file_path.name}")
    except FileNotFoundError:
        logging.error(
            "The 'uncompress' utility is not found. Please install it using Homebrew: 'brew install gzip'"
        )
        print("The 'uncompress' utility is not found. Please install it using Homebrew: 'brew install gzip'")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting {file_path.name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during extraction: {e}")

def download_file(url: str, dest_path: Path):
    """
    Downloads the file from `url` to `dest_path` with a progress bar,
    then extracts the .Z file.
    """
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(dest_path, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=dest_path.name,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        logging.info(f"Downloaded: {dest_path.name}")

        extract_z_file(dest_path)

    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")


async def fetch_download_page_direct_input(page, date_str: str) -> bool:
    """
    Navigates to the downloads page, enters the date (YYYY-MM-DD) directly 
    into the input field via JavaScript, dispatches necessary events,
    clicks the SEARCH button, and waits for the results.
    """
    try:
        await page.goto(DOWNLOADS_URL, timeout=60000)
        logging.info(f"Navigated to {DOWNLOADS_URL}")

        await page.wait_for_selector("#downloadsDatePicker", state='visible', timeout=10000)
        logging.info("Date picker input is visible.")

        await page.evaluate("""
            (date) => {
                const input = document.querySelector('#downloadsDatePicker');
                input.value = date;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """, date_str)
        logging.info(f"Set date via JavaScript: {date_str}")

        await asyncio.sleep(2)
        logging.info("Waited 2 seconds after setting the date.")

        current_value = await page.input_value("#downloadsDatePicker")
        if current_value != date_str:
            logging.warning(f"Input value mismatch: Expected {date_str}, Found {current_value}")
            await page.evaluate("""
                (date) => {
                    const input = document.querySelector('#downloadsDatePicker');
                    input.value = date;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            """, date_str)
            logging.info("Attempted to set the date again via JavaScript.")
            await asyncio.sleep(1)
            current_value = await page.input_value("#downloadsDatePicker")
            if current_value != date_str:
                logging.error(f"Failed to set the date correctly: {date_str}")
                return False
            else:
                logging.info(f"Input value correctly set to {current_value} after retry.")
        else:
            logging.info(f"Input value correctly set to {current_value}")

        await page.click("#downloadsSearchBtn")
        logging.info(f"Clicked SEARCH button for date: {date_str}")

        await asyncio.sleep(3)
        logging.info(f"Waited 3 seconds after clicking SEARCH for date: {date_str}")

        await page.wait_for_selector(".downloads__links", state='visible', timeout=15000)
        logging.info(f"Download links loaded for date: {date_str}")

        return True

    except PlaywrightTimeoutError as e:
        logging.error(f"Timeout while processing date {date_str}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error loading page for date {date_str}: {e}")
        return False

async def parse_market_summary_link(page) -> str:
    """
    Parses the page to find the "Market Summary (Closing)" download link 
    that ends in .Z. Returns the href if found, otherwise an empty string.
    """
    try:
        items = await page.query_selector_all("ul.downloads__links li")
        for item in items:
            text = await item.inner_text()
            if "Market Summary (Closing)" in text:
                a_tag = await item.query_selector("a")
                if not a_tag:
                    continue
                href = await a_tag.get_attribute("href")
                if href and href.endswith(".Z"):
                    if href.startswith("/"):
                        href = BASE_URL + href
                    logging.info(f"Found download link: {href}")
                    return href
        return ""
    except Exception as e:
        logging.error(f"Error parsing download link: {e}")
        return ""


async def fetch_with_retries(page, date_str: str, retries: int = 3, delay: float = 2.0) -> bool:
    """
    Attempts to load the download page with the given date, retrying on failure.
    """
    for attempt in range(1, retries + 1):
        logging.info(f"Attempt {attempt} for date {date_str}")
        success = await fetch_download_page_direct_input(page, date_str)
        if success:
            return True
        else:
            logging.warning(f"Attempt {attempt} failed for date {date_str}. Retrying after {delay} seconds...")
            await asyncio.sleep(delay)
    logging.error(f"All {retries} attempts failed for date {date_str}.")
    return False


async def main():
    print(f"Start Date: {START_DATE}")
    print(f"End Date:   {END_DATE}")
    create_directory(DOWNLOAD_DIR)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        page.on("console", lambda msg: logging.info(f"Console log: {msg.text}"))

        current_date = START_DATE
        while current_date <= END_DATE:
            date_str = current_date.strftime("%Y-%m-%d")

            print(f"Processing date: {date_str} ...")
            logging.info(f"Processing date: {date_str}")

            success = await fetch_with_retries(page, date_str, retries=3, delay=2.0)
            if not success:
                print(f"Skipping {date_str}: Unable to load downloads page.")
                logging.warning(f"Skipping {date_str}: Unable to load downloads page.")
                current_date += datetime.timedelta(days=1)
                await asyncio.sleep(REQUEST_DELAY)
                continue

            link = await parse_market_summary_link(page)
            if not link:
                print(f"Market Summary ZIP not found for {date_str}.")
                logging.warning(f"Market Summary ZIP not found for {date_str}.")
                current_date += datetime.timedelta(days=1)
                await asyncio.sleep(REQUEST_DELAY)
                continue

            filename = link.split("/")[-1]
            dest_path = DOWNLOAD_DIR / filename

            if dest_path.exists():
                print(f"Already downloaded: {filename}")
                logging.info(f"Already downloaded: {filename}")
            else:
                print(f"Found Market Summary (Closing) ZIP: {filename}")
                logging.info(f"Found Market Summary (Closing) ZIP: {filename}")
                download_file(link, dest_path)
                await asyncio.sleep(2)

            current_date += datetime.timedelta(days=1)
            await asyncio.sleep(REQUEST_DELAY)

        await browser.close()
        print("\nAll dates processed. Finished.")
        logging.info("All dates processed. Finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        logging.info("Script interrupted by user.")
