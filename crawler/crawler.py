import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import re

baseURL = "https://eldenring.wiki.fextralife.com"
wikipages = []
visitedURLS = []
notToScrape = ['/Media+%26+Art', '/Build+calculator', '/Interactive+Map']
urls = ["https://eldenring.wiki.fextralife.com/Elden+Ring+Wiki"]
datafile = "../eldenRingWikiText.csv"

with open(datafile, 'w') as csvFile:
    writer = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    while len(urls) != 0:
        print(urls)
        current_url = urls.pop()
        print(current_url)
        response = requests.get(current_url)
        parsedResp = BeautifulSoup(response.content, "html.parser")
        visitedURLS.append(current_url)
        wikipage = {}
        pageTitle = parsedResp.find(id="page-title")
        pageContents = parsedResp.find(id="wiki-content-block")
        wikiLinks = parsedResp.find_all("a", {"class": "wiki_link", "href" : re.compile(r".*")}, )
        for wikiLink in wikiLinks:
            print(wikiLink)
            #print(wikiLink["href"])
            href = wikiLink["href"]
            #print(href)
            #all relevant pages are extensions of baseURL, and should start with /, but should not include files
            if href[0] == "/" and "/file" not in href and href not in notToScrape:
                newUrl = href
                if "https:" not in newUrl and "//eldenring.wiki.fextralife.com" in newUrl:
                    newUrl = "https:" + newUrl
                elif baseURL not in newUrl:
                    newUrl = baseURL+newUrl
                if newUrl not in visitedURLS:
                    urls.append(newUrl)
                #print(href)
        titleFiltered = pageTitle.text.replace(" | Elden Ring Wiki", "")
        wikipage["title"] = titleFiltered
        wikipage["url"] = current_url
        textContents = pageContents.text
        strippedContents = re.sub(r'\n\s*\n', '\\n\\n', textContents)
        strippedContents = strippedContents.replace('\n', '\\n')
        #print(strippedContents)
        wikipage["content"] = strippedContents
        wikipages.append(wikipage)
        writer.writerow(wikipage.values())
#print(wikipages)