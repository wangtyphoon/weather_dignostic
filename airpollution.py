import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from PIL import Image


PAD_TOP = 450
PAD_BOTTOM = 10
PAD_LEFT = 20
PAD_RIGHT = -380
BASE_DIR = Path("air_pollution")


def setup_driver() -> tuple[webdriver.Chrome, WebDriverWait]:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--lang=zh-TW")
    options.add_experimental_option("prefs", {"intl.accept_languages": "zh-TW"})
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 30)
    return driver, wait


def ensure_map_ready(driver: webdriver.Chrome, wait: WebDriverWait):
    container = wait.until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "section.th_taiwan .md_map_ae")
        )
    )
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});", container
    )

    # 等 SVG 完整 render
    wait.until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, "#map_model svg path")) >= 30
    )

    driver.execute_script("""
        const font = '"Noto Sans TC","Microsoft JhengHei","PingFang TC","Heiti TC",sans-serif';
        document.documentElement.style.fontFamily = font;
        document.body.style.fontFamily = font;
    """)
    return container


def sanitize_label(label: str) -> str:
    clean = re.sub(r"\s+", "_", label.strip())
    clean = clean.replace(".", "點")
    return re.sub(r"[^\w\u4e00-\u9fff_]+", "", clean)


def parse_publish_time(driver: webdriver.Chrome) -> datetime:
    txt = driver.find_element(By.ID, "tw_time").text
    m = re.search(r"(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2})", txt)
    if not m:
        raise RuntimeError(f"無法解析發布時間：{txt}")
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y/%m/%d %H:%M")


def reset_to_latest(driver: webdriver.Chrome, wait: WebDriverWait):
    """確保回到最新的時間（重要：避免重新啟動時卡住）"""
    try:
        # 一直點「下一小時」直到按鈕變成 disable，代表已到最新資料
        for _ in range(96):  # safety guard
            btn = driver.find_element(By.ID, "tw_1_after")
            if "disable" in btn.get_attribute("class").split():
                break

            current_stamp = parse_publish_time(driver).strftime("%Y%m%d_%H%M")
            btn.click()
            try:
                wait.until(
                    lambda d: parse_publish_time(d).strftime("%Y%m%d_%H%M") != current_stamp
                    or "disable" in d.find_element(By.ID, "tw_1_after").get_attribute("class")
                )
            except TimeoutException:
                # 若時間軸沒有變化，還是再檢查按鈕是否已禁用
                if "disable" in driver.find_element(By.ID, "tw_1_after").get_attribute("class").split():
                    break
    except Exception:
        pass


def crop_and_save(driver: webdriver.Chrome, container, output_path: Path):

    # 避免元素 rect 尚未穩定
    for _ in range(3):
        rect = container.rect
        if rect["width"] > 200:
            break

    full_screenshot = Image.open(BytesIO(driver.get_screenshot_as_png()))
    left = max(int(rect["x"] - PAD_LEFT), 0)
    top = max(int(rect["y"] - PAD_TOP), 0)
    right = min(int(rect["x"] + rect["width"] + PAD_RIGHT), full_screenshot.width)
    bottom = min(int(rect["y"] + rect["height"] + PAD_BOTTOM), full_screenshot.height)

    cropped = full_screenshot.crop((left, top, right, bottom))
    cropped.save(output_path)


def click_prev_hour(driver: webdriver.Chrome, wait: WebDriverWait, current_stamp: str) -> bool:
    try:
        btn_li = driver.find_element(By.ID, "tw_1_ago")
    except Exception:
        return False

    if "disable" in btn_li.get_attribute("class").split():
        return False

    try:
        btn_li.find_element(By.TAG_NAME, "a").click()
    except Exception:
        return False

    try:
        wait.until(lambda d: parse_publish_time(d).strftime("%Y%m%d_%H%M") != current_stamp)
        return True
    except TimeoutException:
        return False


def download_history_for_tab(driver: webdriver.Chrome, wait: WebDriverWait, tab_label: str, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    seen = set()

    while True:
        current_time = parse_publish_time(driver)
        stamp = current_time.strftime("%Y%m%d_%H%M")

        if stamp in seen:
            break
        seen.add(stamp)

        filepath = output_root / f"{sanitize_label(tab_label)}_{stamp}.png"
        if filepath.exists():
            print(f"已存在 {filepath}，略過並繼續往前...")
            break
        else:
            container = driver.find_element(By.CSS_SELECTOR, "section.th_taiwan .md_map_ae")
            crop_and_save(driver, container, filepath)
            print(f"保存 {tab_label}：{filepath}")

        if not click_prev_hour(driver, wait, stamp):
            break


def run():
    BASE_DIR.mkdir(exist_ok=True)
    driver, wait = setup_driver()

    try:
        driver.get("https://airtw.moenv.gov.tw/")
        ensure_map_ready(driver, wait)

        tab_selector = ".map_targetc .mtg_list a"

        tabs = driver.find_elements(By.CSS_SELECTOR, tab_selector)

        for idx in range(len(tabs)):
            try:
                tabs = driver.find_elements(By.CSS_SELECTOR, tab_selector)
                tab = tabs[idx]
            except StaleElementReferenceException:
                tabs = driver.find_elements(By.CSS_SELECTOR, tab_selector)
                tab = tabs[idx]

            label = tab.text.strip() or tab.get_attribute("title") or f"tab_{idx}"

            tab = driver.find_elements(By.CSS_SELECTOR, tab_selector)[idx]
            wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, tab_selector)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tab)
            try:
                tab.click()
            except Exception:
                driver.execute_script("arguments[0].click();", tab)
            wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, tab_selector)[idx].get_attribute("aria-selected") == "true")

            reset_to_latest(driver, wait)
            download_history_for_tab(driver, wait, label, BASE_DIR / sanitize_label(label))
            if "NO2" in label or "二氧化氮" in label:
                print(f"[{label}] 處理完畢，正在尋找並開啟『測站點位圖』...")
                try:
                    # 使用 XPath 尋找文字包含 '測站點位圖' 的任何元素 (通常是按鈕或 label)
                    station_btn = wait.until(EC.element_to_be_clickable(
                        (By.XPATH, "//*[contains(text(), '測站點位圖')]")
                    ))
                    
                    # 確保按鈕在視野內並點擊
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", station_btn)
                    station_btn.click()
                    
                    print(">>> 成功點擊『測站點位圖』")
                    driver.execute_script("window.scrollBy(0, 400);")
                    # 給予一點時間讓地圖層載入或切換
                    time.sleep(3) 
                    
                    # 選擇性：如果需要等待地圖上出現測站的小圓點，可以加這行
                    # wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, ".site_point")) > 0)
                    
                except TimeoutException:
                    print("!!! 找不到『測站點位圖』按鈕，或按鈕不可點擊。")
                except Exception as e:
                    print(f"!!! 開啟測站點位圖時發生錯誤: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    run()
