import sys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import json
from selenium.common.exceptions import TimeoutException
from queue import Empty
from multiprocessing import Pool, Manager, Lock
import html2text

def setup_chrome_driver():
    """Setup Chrome driver with necessary options for Colab environment"""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-features=NetworkService')
    chrome_options.add_argument('--window-size=1920x1080')
    chrome_options.add_argument('--disable-features=VizDisplayCompositor')

    return webdriver.Chrome(options=chrome_options)




def update_urls(file_path_to_process, extracted_urls, processed_file_path, current_url):
    if os.path.exists(file_path_to_process):
        with open(file_path_to_process, 'r') as file:
            old_urls_to_process = json.load(file)
    else:
        old_urls_to_process = []

    final_urls_to_process = list(set(old_urls_to_process + extracted_urls))

    if os.path.exists(processed_file_path):
        with open(processed_file_path, 'r') as file:
            processed_old_urls = json.load(file)
    else:
        processed_old_urls = []

    processed_old_urls = processed_old_urls + [current_url]
    processed_old_urls = list(set(processed_old_urls))

    with open(processed_file_path, 'w') as file:
        json.dump(processed_old_urls, file, indent=4)


    final_urls_to_process = [url for url in final_urls_to_process if url not in processed_old_urls]
    with open(file_path_to_process, 'w') as file:
        json.dump(final_urls_to_process, file, indent=4)

    return final_urls_to_process, processed_old_urls


def html_to_markdown(url, wait_time=20):

    # Initialize the driver
    driver = setup_chrome_driver()

    try:
        # Load the page
        driver.get(url)

        # Wait for the page to load
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Allow time for JavaScript to execute
        # time.sleep(wait_time)

        # Get the page source
        html_content = driver.page_source

        html_content = re.sub(r'<img\s+[^>]*>', '', html_content)

        # Configure html2text
        h2t = html2text.HTML2Text()
        h2t.body_width = 0  # Disable line wrapping
        h2t.ignore_links = False
        h2t.ignore_images = True
        h2t.ignore_emphasis = False
        h2t.ignore_tables = False

        # Convert to markdown
        markdown_content = h2t.handle(html_content)

        markdown_content = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_content)  # Remove image markdown
        markdown_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)  # Clean up extra newlines


        return markdown_content

    except Exception as e:
        print(f"An error occurred {url}: {str(e)}")
        return None

    finally:
        driver.quit()


def append_to_json_file(data_to_append, json_file_path):
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(data_to_append)

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)




def get_the_url_list(base_url, file_path_to_process, processed_file_path):
  driver = setup_chrome_driver()
  driver.get(base_url)
  extracted_urls = []
  try:
      links = driver.find_elements(By.TAG_NAME, "a")
      for link in links:
          link_url = link.get_attribute("href")
          if link_url is not None:
            link_url = link_url.split('-')[0]
            if "#" in link_url:
              link_url = link_url.split('#')[0]
            if "?" in link_url:
                    link_url = link_url.split('?')[0]

            if (link_url not in extracted_urls) and ("https://help.gohighlevel.com/support/solutions" in link_url) and (link_url != base_url):
                extracted_urls.append(link_url)

  except Exception as e:
      print(f"An error occurred: {e}")

  driver.quit()

  updated_urls, processed_old_urls = update_urls(file_path_to_process, extracted_urls, processed_file_path, base_url)

  output_dir = "content/crawl/scraped_content"
  os.makedirs(output_dir, exist_ok=True)

  url = base_url

  suc_url = None
  failed_url = None

  try:
      if '/page/' in url:
        filename = f"{url.split('/')[-3]}_{url.split('/')[-2]}_{url.split('/')[-1]}.md"

      else:
        filename = url.split('/')[-1] + '.md'

      output_path = os.path.join(output_dir, filename)

      markdown_content = html_to_markdown(url)


      if markdown_content:
          with open(output_path, 'w', encoding='utf-8') as f:
              f.write(markdown_content)
          print(f"Successfully saved content to {output_path}")

          suc_url = url

          append_to_json_file({'url': url, 'file_path': output_path}, "content/crawl/md_url_mapping.json")




  except Exception as e:
      print(f"Failed to process {url}: {str(e)}")
      failed_url = url

  return suc_url, failed_url, updated_urls, processed_old_urls


def run(base_url, file_path_to_process, processed_file_path):
  suc_url_list = []
  failed_url_list = []
  c = 0
  while True:
    suc_url, failed_url, updated_urls, processed_old_urls = get_the_url_list(base_url, file_path_to_process, processed_file_path)
    if suc_url is not None:
      suc_url_list.append(suc_url)
    if failed_url is not None:
      failed_url_list.append(failed_url)


    if len(updated_urls) > 0:
      base_url = updated_urls[0]
      if base_url not in processed_old_urls:
        continue
      else:
        break
    else:
        break
  return suc_url_list, failed_url_list

file_path_to_process = "content/crawl/extracted_urls.json"
processed_file_path = "content/crawl/processed_urls.json"
base_url = "https://help.gohighlevel.com/support/solutions"

os.makedirs("content/crawl", exist_ok=True)
os.makedirs("content/crawl/scraped_content", exist_ok=True)

suc_url_list, failed_url_list = run(base_url, file_path_to_process, processed_file_path)
