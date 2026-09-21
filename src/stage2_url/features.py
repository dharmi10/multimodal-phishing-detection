import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

def tokenize_url(url):
    return tokenizer.tokenize(url.lower())

def has_ip_address(url):
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    return bool(re.search(ip_pattern, url))

def count_slashes(url):
    return url.count('/')

def has_port_number(url):
    url_no_scheme = re.sub(r'^https?://', '', url)
    port_pattern = r':\d{2,5}'
    return bool(re.search(port_pattern, url_no_scheme))

def has_at_symbol(url):
    return '@' in url

def count_hyphens(url):
    return url.count('-')

def url_length(url):
    return len(url)

def domain_length(url):
    try:
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        return len(parsed.netloc)
    except ValueError:
        return 0

def extract_lexical_features(url):
    return {
        'has_ip_address': has_ip_address(url),
        'slash_count': count_slashes(url),
        'has_port_number': has_port_number(url),
        'has_at_symbol': has_at_symbol(url),
        'hyphen_count': count_hyphens(url),
        'url_length': url_length(url),
        'domain_length': domain_length(url),
    }

def fetch_resource(url, timeout=5):
    """
    Fetch a URL once and report what came back.

    Returns None when the request failed (down, blocked, timeout), otherwise a
    dict with the declared content type alongside both the decoded text and the
    raw bytes, so the caller can decide which one is meaningful.

    The Content-Type header was already being received and thrown away; reading
    it costs nothing extra on the wire. The timeout is unchanged at 5s.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException:
        return None

    declared = response.headers.get('Content-Type', '') or ''
    content_type = declared.split(';', 1)[0].strip().lower()

    return {
        'content_type': content_type,
        'is_html': content_type in ('text/html', 'application/xhtml+xml'),
        'is_image': content_type.startswith('image/'),
        'text': response.text,
        'content': response.content,
    }


def fetch_html(url, timeout=5):
    """Backwards-compatible wrapper: just the decoded body, or None."""
    resource = fetch_resource(url, timeout=timeout)
    return None if resource is None else resource['text']

def get_link_ratio(soup, base_url):
    base_domain = urlparse(base_url).netloc
    internal = 0
    external = 0
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        link_domain = urlparse(href).netloc
        if link_domain == '' or link_domain == base_domain:
            internal += 1
        else:
            external += 1
    total = internal + external
    ratio = external / total if total > 0 else 0
    return internal, external, ratio

def has_hidden_iframe(soup):
    for iframe in soup.find_all('iframe'):
        width = str(iframe.get('width', '')).strip()
        height = str(iframe.get('height', '')).strip()
        style = str(iframe.get('style', '')).lower()
        if width in ('0', '1') or height in ('0', '1') or 'display:none' in style.replace(' ', ''):
            return True
    return False

def has_suspicious_form(soup, base_url):
    base_domain = urlparse(base_url).netloc
    for form in soup.find_all('form'):
        action = form.get('action', '').strip()
        action_domain = urlparse(action).netloc
        if action_domain != '' and action_domain != base_domain:
            return True
    return False

def blocks_right_click(html_text):
    suspicious_snippets = ['event.button==2', 'oncontextmenu', 'contextmenu']
    html_lower = html_text.lower()
    return any(snippet in html_lower for snippet in suspicious_snippets)

def analyze_html(html_text, url):
    """
    The four red flags, computed from an already-fetched body.

    Split out of check_website_html so a caller that has done the fetch itself
    (to look at the Content-Type first) does not have to fetch a second time.
    The checks themselves are unchanged.
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    internal, external, ext_ratio = get_link_ratio(soup, url)
    return {
        'external_link_ratio': round(ext_ratio, 2),
        'has_hidden_iframe': has_hidden_iframe(soup),
        'has_suspicious_form': has_suspicious_form(soup, url),
        'blocks_right_click': blocks_right_click(html_text),
    }


def check_website_html(url):
    html_text = fetch_html(url)
    if html_text is None:
        return None
    return analyze_html(html_text, url)