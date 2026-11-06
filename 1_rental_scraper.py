from datetime import datetime
import json
import os
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# Define one or multiple search areas to scrape simultaneously.
# Combining areas aggregates and deduplicates listing URLs before fetching details.

SEARCH_AREAS = [
    {
        "name": "Vaughan (19km)",
        "base_path": "https://www.kijiji.ca/b-apartments-condos/markham-york-region",
        "cat_id": "c37l1700274",
        "query_params": "ll=43.82622809999999%2C-79.5561668&radius=19.0&view=list",
    }
]

# Scraping & Pagination Limits
MAX_PAGES_PER_AREA = 102  # Kijiji caps results per query at 100-102 pages
HEADLESS_MODE = False

# Output Configuration
OUTPUT_FILE_PREFIX = "kijiji_rental_ads_canonical_vaughan"

# Browser Navigation Settings
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
VIEWPORT = {"width": 1920, "height": 1080}
INDEX_TIMEOUT_MS = 45000
DETAIL_TIMEOUT_MS = 30000
INDEX_SLEEP_SEC = 1.0
DETAIL_SLEEP_SEC = 1.2


# ==============================================================================
# PARSING & EXTRACTION HELPERS
# ==============================================================================
def extract_mls_number(html_content, soup=None):
    """
    Extracts the MLS number from HTML content, JSON payloads, or DOM text.
    """
    if not html_content:
        return None

    mls_patterns = [
        r"MLS®?\s*#\s*([A-Za-z0-9]+)",
        r'"mlsNumber"\s*:\s*"([^"]+)"',
        r'"mls_number"\s*:\s*"([^"]+)"',
        r"MLS\s*#?\s*:\s*([A-Za-z0-9]+)",
    ]

    for pattern in mls_patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if soup:
        text_content = soup.get_text(separator=" ")
        match = re.search(r"MLS®?\s*#\s*([A-Za-z0-9]+)", text_content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_location_scores(soup):
    """
    Extracts walkscore, transitscore, and bikescore in a single DOM pass.
    Targets containers where one element holds the label and its sibling/parent holds the value.
    """
    scores = {"walkscore": None, "transitscore": None, "bikescore": None}

    if not soup:
        return scores

    # Map target DOM labels ("walk score", "transit score", "bike score") to dictionary keys
    label_mapping = {
        "walk score": "walkscore",
        "transit score": "transitscore",
        "bike score": "bikescore",
    }

    label_nodes = soup.find_all(
        lambda el: el.name in ["p", "span", "div", "dt"]
        and el.get_text(strip=True).lower() in label_mapping
    )

    for label_node in label_nodes:
        score_key = label_mapping[label_node.get_text(strip=True).lower()]
        if scores[score_key]:
            continue

        # 1. Search parent wrapper for adjacent numeric text element
        parent = label_node.find_parent(["div", "li", "section", "dl", "tr"])
        if parent:
            for el in parent.find_all(["p", "span", "div", "dd"]):
                txt = el.get_text(strip=True)
                if txt.isdigit() and el != label_node:
                    scores[score_key] = txt
                    break

        # 2. Next sibling fallback
        if not scores[score_key]:
            sibling = label_node.find_next_sibling()
            if sibling:
                sibling_txt = sibling.get_text(strip=True)
                if sibling_txt.isdigit():
                    scores[score_key] = sibling_txt

    return scores


def extract_kijiji_canonical_fields(html_content, url=""):
    """
    Parses a Kijiji listing HTML page and extracts Title, Price, Address,
    LocationCoordinates, Date Posted, MLS #, walkscore/transitscore/bikescore,
    and all dynamic attributes using Next.js Apollo State (__NEXT_DATA__),
    JSON-LD schema, and HTML DOM fallbacks.
    """
    soup = BeautifulSoup(html_content, "html.parser") if html_content else None

    record = {
        "Title": None,
        "Price($)": None,
        "Address": None,
        "LocationCoordinates": None,
        "Date Posted": None,
        "url": url,
        "mls_number": None,
        "walkscore": None,
        "transitscore": None,
        "bikescore": None,
    }

    if not html_content:
        return record

    # 1. Extract MLS number & Location Scores from DOM
    record["mls_number"] = extract_mls_number(html_content, soup)

    if soup:
        dom_scores = extract_location_scores(soup)
        for score_key, score_val in dom_scores.items():
            if score_val:
                record[score_key] = score_val

    # 2. Parse JSON-LD Schema (Title, Price, Address & Geo Coordinates)
    if soup:
        for tag in soup.find_all("script", type="application/ld+json"):
            if not tag.string:
                continue
            try:
                data = json.loads(tag.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if isinstance(item, dict) and item.get("@type") in [
                        "SingleFamilyResidence",
                        "Apartment",
                        "Product",
                        "RealEstateListing",
                        "Offer",
                    ]:
                        if not record["Title"] and item.get("name"):
                            record["Title"] = item.get("name")

                        offers = item.get("offers")
                        if isinstance(offers, dict) and offers.get("price"):
                            try:
                                record["Price($)"] = float(offers["price"])
                            except (ValueError, TypeError):
                                pass

                        addr = item.get("address")
                        if isinstance(addr, dict):
                            street = addr.get("streetAddress")
                            city = addr.get("addressLocality")
                            record["Address"] = street or city
                        elif isinstance(addr, str):
                            record["Address"] = addr

                        geo = item.get("geo")
                        if isinstance(geo, dict) and not record["LocationCoordinates"]:
                            lat = geo.get("latitude")
                            lng = geo.get("longitude")
                            if lat is not None and lng is not None:
                                record["LocationCoordinates"] = f"{lat}, {lng}"
            except Exception:
                continue

    # 3. Parse Next.js Apollo State (__NEXT_DATA__)
    if soup:
        next_script = soup.find("script", id="__NEXT_DATA__")
        if next_script and next_script.string:
            try:
                next_data = json.loads(next_script.string)
                apollo_state = (
                    next_data.get("props", {})
                    .get("pageProps", {})
                    .get("__APOLLO_STATE__", {})
                )

                for key, obj in apollo_state.items():
                    if isinstance(obj, dict) and (
                        obj.get("__typename") == "Listing" or "RealEstateListing" in key
                    ):
                        record["Title"] = record["Title"] or obj.get("title")
                        record["Date Posted"] = (
                            obj.get("activationDate")
                            or obj.get("sortingDate")
                            or record["Date Posted"]
                        )

                        if obj.get("price") and isinstance(obj["price"], dict):
                            record["Price($)"] = (
                                record["Price($)"] or obj["price"].get("amount")
                            )

                        loc = obj.get("location", {})
                        if isinstance(loc, dict):
                            if not record["Address"]:
                                parts = [
                                    loc.get("streetAddress"),
                                    loc.get("city"),
                                    loc.get("province"),
                                    loc.get("postalCode"),
                                ]
                                record["Address"] = ", ".join([p for p in parts if p])

                            if not record["LocationCoordinates"]:
                                coords = loc.get("coordinates")
                                if isinstance(coords, dict):
                                    if "latitude" in coords and "longitude" in coords:
                                        record["LocationCoordinates"] = (
                                            f"{coords.get('latitude')}, {coords.get('longitude')}"
                                        )
                                    elif (
                                        "__ref" in coords
                                        and coords["__ref"] in apollo_state
                                    ):
                                        ref_obj = apollo_state[coords["__ref"]]
                                        if isinstance(ref_obj, dict):
                                            lat = ref_obj.get("latitude")
                                            lng = ref_obj.get("longitude")
                                            if lat is not None and lng is not None:
                                                record["LocationCoordinates"] = (
                                                    f"{lat}, {lng}"
                                                )

                        # Dynamic Attributes
                        raw_attrs = obj.get("attributes", {})
                        attr_list = []
                        if isinstance(raw_attrs, dict):
                            attr_list = raw_attrs.get("all", []) or raw_attrs.get("list", [])
                        elif isinstance(raw_attrs, list):
                            attr_list = raw_attrs

                        for attr in attr_list:
                            if isinstance(attr, dict):
                                c_name = (
                                    attr.get("canonicalName")
                                    or attr.get("machineName")
                                    or attr.get("name")
                                )
                                if not c_name:
                                    continue

                                c_header = c_name.strip().lower()
                                vals = []

                                if isinstance(attr.get("values"), list) and attr["values"]:
                                    vals.extend([str(v) for v in attr["values"] if v is not None])
                                elif isinstance(attr.get("canonicalValues"), list) and attr["canonicalValues"]:
                                    vals.extend([str(v) for v in attr["canonicalValues"] if v is not None])

                                loc_val = attr.get("localizedValue") or attr.get("value")
                                if loc_val and str(loc_val) not in vals:
                                    vals.append(str(loc_val))

                                if vals:
                                    record[c_header] = ", ".join(vals)

                        # Extract Scores from Apollo State
                        scores = obj.get("locationScores") or obj.get("scores")
                        if isinstance(scores, dict):
                            if "__ref" in scores and scores["__ref"] in apollo_state:
                                scores = apollo_state[scores["__ref"]]

                            if isinstance(scores, dict):
                                w_score = scores.get("walkScore")
                                t_score = scores.get("transitScore")
                                b_score = scores.get("bikeScore")

                                if w_score is not None and str(w_score).strip().isdigit():
                                    record["walkscore"] = record["walkscore"] or str(w_score).strip()
                                if t_score is not None and str(t_score).strip().isdigit():
                                    record["transitscore"] = record["transitscore"] or str(t_score).strip()
                                if b_score is not None and str(b_score).strip().isdigit():
                                    record["bikescore"] = record["bikescore"] or str(b_score).strip()
            except Exception:
                pass

    # 4. HTML DOM Fallback for Title
    if soup and not record["Title"]:
        h1 = soup.find("h1")
        if h1:
            record["Title"] = h1.get_text(strip=True)

    return record


# ==============================================================================
# SCRAPER CLASS
# ==============================================================================
class KijijiGTARentalScraper:
    def __init__(self, search_areas=None, headless=HEADLESS_MODE, max_pages=MAX_PAGES_PER_AREA):
        self.search_areas = search_areas or SEARCH_AREAS
        self.headless = headless
        self.max_pages = min(102, max_pages)

    def build_page_url(self, area_config, page_num):
        """Constructs valid Kijiji pagination URLs matching platform route syntax."""
        base_path = area_config["base_path"]
        cat_id = area_config["cat_id"]
        query_params = area_config["query_params"]

        if page_num == 1:
            return f"{base_path}/{cat_id}?{query_params}"
        return f"{base_path}/page-{page_num}/{cat_id}?{query_params}"

    def save_progress_to_csv(self, extracted_data, output_filename):
        """Helper to re-index and save accumulated records to CSV incrementally."""
        if not extracted_data:
            return None

        df = pd.DataFrame(extracted_data)

        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"], keep="last")

        core_cols = ["Title", "Price($)", "Address", "LocationCoordinates", "Date Posted"]
        tail_cols = ["url", "mls_number", "walkscore", "transitscore", "bikescore"]

        dynamic_cols = sorted(
            [col for col in df.columns if col not in core_cols and col not in tail_cols]
        )

        final_cols = core_cols + dynamic_cols + tail_cols
        df = df.reindex(columns=final_cols)
        df.to_csv(output_filename, index=False, encoding="utf-8-sig")
        return df

    def scrape(self, output_filename=None):
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{OUTPUT_FILE_PREFIX}_{timestamp}.csv"

        extracted_data = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport=VIEWPORT,
            )
            page = context.new_page()
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

            listing_urls = []
            seen_urls = set()

            # Step 1: Collect listing URLs across all search areas simultaneously
            for area in self.search_areas:
                area_name = area.get("name", "Target Area")
                print(f"\n--- Gathering Listings for Area: {area_name} ---")

                for p_num in range(1, self.max_pages + 1):
                    target_url = self.build_page_url(area, p_num)
                    print(f"[{area_name}] Index Page {p_num}/{self.max_pages}: {target_url}")

                    try:
                        page.goto(target_url, wait_until="domcontentloaded", timeout=INDEX_TIMEOUT_MS)
                        time.sleep(INDEX_SLEEP_SEC)
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        time.sleep(0.5)

                        soup = BeautifulSoup(page.content(), "html.parser")
                        links = soup.find_all("a", href=re.compile(r"/v-apartments-condos/"))

                        found_on_page = 0
                        for a in links:
                            href = a.get("href")
                            if href:
                                if not href.startswith("http"):
                                    href = f"https://www.kijiji.ca{href}"
                                clean_url = href.split("?")[0]
                                if clean_url not in seen_urls:
                                    seen_urls.add(clean_url)
                                    listing_urls.append(clean_url)
                                    found_on_page += 1

                        print(f"  -> Discovered {found_on_page} new unique listings (Total: {len(listing_urls)})")

                        if found_on_page == 0:
                            print(f"[{area_name}] No new listings on page {p_num}. Reached pagination end.")
                            break

                    except Exception as e:
                        print(f"[Error] Index page {p_num} failed for area {area_name}: {e}")
                        break

            # Step 2: Parse detail pages and update CSV incrementally
            print(f"\n[Details] Scraping {len(listing_urls)} collected unique listing URLs...")
            for idx, url in enumerate(listing_urls, 1):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=DETAIL_TIMEOUT_MS)
                    time.sleep(DETAIL_SLEEP_SEC)
                    record = extract_kijiji_canonical_fields(page.content(), url)
                    extracted_data.append(record)

                    # Save progress after each detail page
                    self.save_progress_to_csv(extracted_data, output_filename)

                    title_preview = record["Title"][:100] if record["Title"] else "No Title"
                    price_preview = f"${record['Price($)']}" if record["Price($)"] else "No Price"
                    mls_preview = f"MLS#: {record['mls_number']}" if record["mls_number"] else "No MLS#"
                    scores_preview = f"Walk: {record['walkscore']} | Transit: {record['transitscore']} | Bike: {record['bikescore']}"
                    print(f"[{idx}/{len(listing_urls)}] Scraped & Saved: {title_preview} | {price_preview} | {mls_preview} | {scores_preview}")
                except Exception as e:
                    print(f"[Error] Detail page failed ({url}): {e}")

            browser.close()

        # Step 3: Final output summary
        if extracted_data:
            final_df = self.save_progress_to_csv(extracted_data, output_filename)
            print("\n=======================================================")
            print(f"Successfully finished scrape ({final_df.shape[0]} rows, {final_df.shape[1]} columns)")
            print(f"File saved at: {output_filename}")
            print("=======================================================\n")
            print("DataFrame Sample Preview:")
            print(final_df.head())
            return final_df
        else:
            print("\n[Warning] No listings parsed.")
            return pd.DataFrame()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_FILE_PREFIX}_{timestamp}.csv"

    scraper = KijijiGTARentalScraper(
        search_areas=SEARCH_AREAS,
        headless=HEADLESS_MODE,
        max_pages=MAX_PAGES_PER_AREA,
    )
    scraper.scrape(output_filename)