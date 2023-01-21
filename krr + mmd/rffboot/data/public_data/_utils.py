import requests


def _download_url(folder, url):
    r = requests.get(url, allow_redirects=True)
    download_data = folder / url.rsplit("/", maxsplit=1)[1]
    with open(download_data, "wb") as f:
        f.write(r.content)
    return download_data
