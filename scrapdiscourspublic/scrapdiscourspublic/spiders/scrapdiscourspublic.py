import scrapy
#from scrapdiscourspublic.items import ScrapdiscourspublicItem


class ScrapDiscoursPublic(scrapy.Spider):
    name = "sdp"

    def start_requests(self):
        step = 100
        total = 55658
        start = [i*step for i in range(total//step+1)]

        urls = [
'http://www.vie-publique.fr/rechercher/recherche.php?query=&date=&dateDebut=&dateFin=&b='+str(s)+'&skin=cdp&replies='+str(step)+'&filter=&typeloi=&auteur=&filtreAuteurLibre=&typeDoc=f/vp_type_discours/declaration&source=&sort=&q=' for s in start
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        uglyurls = response.xpath('.//*[@id="subcontent"]/ul/li[*]/p[1]/a/@href').extract()
        cleanurls = [ug.strip(' \n') for ug in uglyurls]
        for cleanurl in cleanurls:
            yield scrapy.Request(url=cleanurl, callback=self.discours)

    def discours(self, response):
        item = {}
        print(response.url)
        item['title'] = response.xpath('.//*[@id="content"]/div[1]/div[@class="title_article"]/h2/text()').extract()
        item['personalite'] = response.xpath('.//*[@id="content"]/div[1]/div[@class="article"]/p[1]/strong/following-sibling::text()').extract()
        item['fonction'] = response.xpath('.//*[@id="content"]/div[1]/div[@class="article"]/p[2]/text()').extract()
        item['circonstances'] = response.xpath('.//*[@id="content"]/div[1]/div[@class="article"]/p[3]/strong/following-sibling::text()').extract()
        item['discours'] = response.xpath('.//*[@id="content"]/div[1]/p/text()').extract()
        yield item