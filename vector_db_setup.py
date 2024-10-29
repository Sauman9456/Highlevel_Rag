import os
import json
import re
from datetime import datetime
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma


os.environ["OPENAI_API_KEY"] = "Add your key"
os.environ["COHERE_API_KEY"] = "Add your key"

def split_text_by_first_newline_hashtag(text):
    # Use regular expression to find the first occurrence of '#' at the start of a new line
    match = re.search(r'(?m)^\#', text)

    if match:
        # Split the text into two parts
        first_part = text[:match.start()].strip()
        second_part = text[match.start():].strip()
        return second_part
    else:
        # If no such pattern is found, return the entire text as the first part
        return text.strip()

def remove_toc_urls(text):
    # Define regex pattern to match URLs
    url_pattern = r"https?://\S+"

    # Remove URLs only under the "TABLE OF CONTENTS"
    table_of_contents_section = re.search(r"TABLE OF CONTENTS(.*?)(?=\n#|\Z)", text, re.DOTALL)

    if table_of_contents_section:
        content_to_modify = table_of_contents_section.group(1)
        cleaned_content = re.sub(url_pattern, '', content_to_modify)
        cleaned_text = text.replace(content_to_modify, cleaned_content)
    else:
        cleaned_text = text

    # cleaned_text = cleaned_text.replace("Was this article helpful?", "")
    return cleaned_text


def process_text_with_ts(text):
    modified_pattern = r"^Modified on:\s*(.*)"

    lines = text.split('\n')
    converted_timestamp = None
    new_lines = []

    for line in lines:
        match = re.match(modified_pattern, line)
        if match and not converted_timestamp:
            try:
                original_time = datetime.strptime(match.group(1), "%a, %d %b, %Y at %I:%M %p")
                converted_timestamp = original_time.strftime("%d-%m-%Y %H:%M:%S")
            except ValueError:
                pass
            continue
        new_lines.append(line)

    processed_text = '\n'.join(new_lines).strip()

    return processed_text, converted_timestamp


def clean_markdown_nested_list(content):
    """
    Clean markdown content by removing duplicates and fixing nested list formatting.

    Args:
        content (str): Input markdown content

    Returns:
        str: Cleaned markdown content
    """
    # Split content into lines
    lines = content.strip().split('\n')

    # Store processed lines and their lowercase versions for comparison
    processed_lines = []
    seen_content = set()

    # Track list hierarchy and state
    current_parent = None
    parent_items = []
    in_nested_list = False

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Handle empty lines
        if not line:
            if processed_lines and processed_lines[-1].strip():
                processed_lines.append('')
            i += 1
            continue

        # Handle headers
        if line.startswith('#'):
            current_parent = None
            parent_items = []
            in_nested_list = False
            if line.lower() not in seen_content:
                processed_lines.append(line)
                seen_content.add(line.lower())
            i += 1
            continue

        # Handle list items
        list_match = re.match(r'^[*+-]\s+(.+)$', line)
        if list_match:
            list_content = list_match.group(1)

            # Check if this is a parent list item (ends with ":")
            if list_content.endswith(':'):
                if list_content.lower() not in seen_content:
                    processed_lines.append(f"* {list_content}")
                    seen_content.add(list_content.lower())
                    current_parent = list_content
                    parent_items = []
                    in_nested_list = True
            else:
                # Regular list item
                if list_content.lower() not in seen_content:
                    processed_lines.append(f"* {list_content}")
                    seen_content.add(list_content.lower())
            i += 1
            continue

        # Handle potential nested list items
        if in_nested_list and current_parent and not line.startswith('#'):
            # Check if this line should be part of the nested list
            if (line.lower() not in seen_content and
                line.lower() not in (item.lower() for item in parent_items)):
                processed_lines.append(f"  * {line}")
                seen_content.add(line.lower())
                parent_items.append(line)
            i += 1
            continue

        # Handle regular text content
        if line.lower() not in seen_content:
            processed_lines.append(line)
            seen_content.add(line.lower())
            current_parent = None
            in_nested_list = False

        i += 1

    # Clean up consecutive empty lines
    result = []
    empty_line_count = 0
    for line in processed_lines:
        if not line.strip():
            empty_line_count += 1
            if empty_line_count <= 2:  # Keep up to 2 consecutive empty lines
                result.append(line)
        else:
            empty_line_count = 0
            result.append(line)

    return '\n'.join(result)


def clean_markdown_file(file_content):
  #remove backslashes (\) that are next to numbers
  file_content = re.sub(r'(?<=\d)\\', '', file_content)
  file_content = split_text_by_first_newline_hashtag(file_content)
  file_content = file_content.split("javascript:print()")[0].split("# **Related Articles")[0].split("Articles in this folder -")[0].split("Was this article helpful?")[0]
  # file_content = clean_markdown_nested_list(file_content)
  file_content = remove_toc_urls(file_content)
  file_content, file_timestamp = process_text_with_ts(file_content)
  file_content = re.sub(r'\n\s*\n+', '\n\n', file_content)
  return file_content, file_timestamp


def extract_index(text):
    lines = text.strip().split("\n")
    index = []

    title = lines[0].strip()
    index.append(f"{title}")

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("#"):
            index_line = "#"+line
            index_line_ = "##"+line
            if index_line in index:
              index.remove(index_line)
            elif line in index:
              index.remove(line)
            elif index_line_ in index:
              index.remove(index_line_)

            index.append(index_line)

    return "\n".join(index), title[1:].strip()


def combine_strings(strings, max_length=3000):
    combined_list = []
    current_combination = ""

    for s in strings:
        if len(s) > max_length:
            # If a single string is longer than max_length, add the previous combination if exists
            if current_combination:
                combined_list.append(current_combination)
                current_combination = ""
            # Add the large string directly
            combined_list.append(s)
        else:
            # Check if adding this string would exceed max_length
            if len(current_combination) + len(s) > max_length:
                # Add the current combination to the list and start a new combination
                combined_list.append(current_combination)
                current_combination = s
            else:
                # Add the string to the current combination
                current_combination += s

    # Add any remaining combination
    if current_combination:
        combined_list.append(current_combination)

    return combined_list


def split_long_section(section):
    h3_pattern = re.compile(r'^##\s+.*$', re.MULTILINE)
    h3_matches = list(h3_pattern.finditer(section))
    if not h3_matches:
        return [section]
    subsections = []
    start = 0
    current_length = 0

    for match in h3_matches:
        if current_length + (match.start() - start) > 3000:
            # If adding this subsection would exceed 10000 characters, split here
            subsections.append(section[start:match.start()].strip())
            start = match.start()
            current_length = 0
        else:
            current_length += match.start() - start

    # Add the last subsection
    subsections.append(section[start:].strip())

    return subsections




def split_markdown_by_headings(markdown_text):
    heading_pattern = re.compile(r'^(#{1})\s+.*$', re.MULTILINE)

    headings = heading_pattern.finditer(markdown_text)

    positions = [match.start() for match in headings]
    positions.append(len(markdown_text))
    sections = []
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        section = markdown_text[start:end].strip()
        if section.startswith('# ') and len(section) > 3000:
            subsections = split_long_section(section)
            sections.extend(subsections)
        else:
            sections.append(section)

    sections = combine_strings(sections)
    return sections


class GetSummary(BaseModel):
    summary: str = Field(description="Summary of the given document")

def generate_summary(doc):
  client = instructor.from_openai(OpenAI())
  system_prompt = f"""
Your task is to generate a concise summary of the provided document in under 100 words. This summary will help filter the document based on the user's query, assessing whether the document contains enough information to answer it. Ensure all critical details are covered.
Make summary is under 100 words.

# Document
----------------------------------------------------------------------------------------------------------------
{doc}
----------------------------------------------------------------------------------------------------------------


Summary:
"""
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{'role': 'system', 'content': system_prompt}],
      response_model=GetSummary,
      temperature=0
  )
  return response.summary


def langchain_doc(file_content, url, file_index, title, file_timestamp):
  docs = []

  page_content_token_count = len(file_content)//4
  page_index_token_count = len(file_index)//4

  if file_content is None:
    file_content = ""
    page_content_token_count = 0

  if file_index is None:
    file_index = ""
    page_index_token_count = 0

  if file_timestamp is None:
    file_timestamp = ""
  if url is None:
    url = ""


  if len(file_content) > 4000:
    file_contents = split_markdown_by_headings(file_content)
  else:
    file_contents = [file_content]

  for section in file_contents:
    if len(file_contents)<2:
      section_index = ""
      section_title = ""
      section_index_token_count = 0
      section_content_token_count = 0
    else:
      section_index, section_title = extract_index(section)
      section_index_token_count = len(section_index)//4
      section_content_token_count = len(section)//4

    if section is None:
      section = ""
      section_content_token_count = 0
    if section_index is None:
      section_index = ""
      section_index_token_count = 0
    if section_title is None:
      section_title = ""

    summary = ""
    if section is not None:
     summary =  generate_summary(section)

    doc =  Document(page_content=section, metadata={"page_index": file_index, "url": url,
                                                    "modified_on": file_timestamp, "title": title,
                                                    "page_content_token_count":page_content_token_count,
                                                    "page_index_token_count": page_index_token_count,
                                                    "section_index": section_index,
                                                    "section_title": section_title,
                                                    "section_content_token_count" : section_content_token_count,
                                                    "section_index_token_count": section_index_token_count,
                                                    "section_summary":summary})
    docs.append(doc)
  return docs

def langchain_doc_loader(mapping_path):
  docs = []
  total_index_tokens = 0
  total_content_tokens = 0
  titles = ""

  combine_file_content = ""

  with open(mapping_path, 'r') as file:
    md_url_mapping = json.load(file)

  md_url_mapping = [dict(t) for t in {frozenset(d.items()) for d in md_url_mapping}]

  for path in md_url_mapping:
    if "/articles" in path['url']:
      with open(path['file_path'], 'r') as file:
        # Read the content of the file
        file_content = file.read()
      file_content, file_timestamp = clean_markdown_file(file_content)
      combine_file_content = combine_file_content + "\n\n----------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx------------------------------------\n\n" + file_content
      file_index, title = extract_index(file_content)

      if len(title) > 25:
        if title not in titles:
          titles = titles + "# " + title + "\n\n"

      doc = langchain_doc(file_content, path['url'], file_index, title, file_timestamp)


      docs.extend(doc)
  return docs, titles, combine_file_content

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small",)

docs, titles, combine_file_content = langchain_doc_loader(mapping_path = "content/crawl/md_url_mapping_2.json")


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory = "/content/crawl/high_level_support_solution_chroma_langchain_db")
vectorstore_retreiver = vectorstore.as_retriever()