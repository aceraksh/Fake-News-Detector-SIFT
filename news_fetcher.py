import requests

API_KEY = "9574ead008e44845ba97d56d2686ee3c"


def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=5&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    articles = []

    if data["status"] != "ok":
        print("API Error:", data)
        return []

    for article in data["articles"]:
        title = article["title"] if article["title"] else ""
        description = article["description"] if article["description"] else ""
        text = f"{title} {description}"
        articles.append(text)

    return articles
