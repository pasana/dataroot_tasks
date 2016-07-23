# -*- coding: utf-8 -*-
import scrapy
import time
import re
from spider_1000realty.items import Spider1000RealtyItem

ADDR_RE = [
ur"(?P<street>\w*\W+), (?P<house_number>[буд.майд№ ]*\d+[/а-яА-Я\\-]*\d*[а-яА-я/]*), (?P<area>\W+\w*), (?P<city>Киев)", 
ur"(?P<street>\w*\W+), (?P<house_number>[буд.майд№ ]*\d+[/а-яА-Я\\-]*\d*[а-яА-я/]*), (?P<area>\W+\w*), (?P<district>\W+\w*), (?P<city>Киев)", 
ur"(?P<street>\w*\W+), (?P<house_number>[буд.майд№ ]*\d+[/а-яА-Я\\-]*\d*[а-яА-я/]*), (?P<city>Киев)", 
ur"(?P<street>\w*\W+), (?P<area>\W+\w*), (?P<district>\W+\w*), (?P<city>Киев)"
]

class SaleSpider(scrapy.Spider):
    custom_settings = {
#        'CONCURRENT_REQUESTS': 1,
#        'CONCURRENT_ITEMS': 1,
#        'DOWNLOAD_DELAY': 1,
        'USER_AGENT': 'pasana',
   }
    name = "sale"
    allowed_domains = ["100realty.com"]
    start_urls = (
        'http://100realty.ua/realty_search/apartment/rent/ro_2/cur_3/kch_2',
        'http://100realty.ua/realty_search/apartment/sale/ro_2/cur_3/kch_2',
#	'http://100realty.ua/object/372343257',
    )

    def parse(self, response):
        total = response.xpath('//div[@id="realty-search-informer"]/div[@class="content"]/text()').extract()[0].split()[-1]
        pages = int(total) / 50
        for i in range(0, pages):
            url = response.urljoin("kch_2?page=%d"%i)
            yield scrapy.Request(url, callback=self.parse_page, dont_filter=True)

    def parse_page(self, response):
        results = response.xpath('//div[@id="realty-search-results"]/div/div')
        for r in results:
            try:
                url = r.xpath('div/div/div[@class="object-address"]/a/@href').extract()[0]
                #print url
                yield scrapy.Request("http://100realty.ua/"+url, callback=self.parse_object, dont_filter=True)
            except:
                pass

    def parse_object(self, response):
        item = Spider1000RealtyItem()
        t = response.xpath('//div[@class="breadcrumb"]/a[2]/@href').extract()[0].split('/')[-1]
        if t == 'rent':
            item['type'] = 1
        if t == 'sale':
            item['type'] = 2
        context = response.xpath('//div[@id="object-information"]')
        price = context.xpath('div[@id="object-price"]/div[@class="price"]/text()').extract()[0]
        if t == 'rent':
            pay_period = context.xpath('div[@id="object-price"]/div[@class="pay-period"]/text()').extract()[0][2:]
            if pay_period == u"Месяц":
                item['price'] = ''.join(re.findall(r"\d+", price))
            elif pay_period == u"Год":
                item['price'] = str(float(price.replace(' ', '')[:-7])/12)
        if t == 'sale':
                item['price'] = ''.join(re.findall(r"\d+", price))
        try:
            item['description'] = context.xpath('div/div/div[@id="object-total-info"]/text()').extract()
        except:
            pass
        #address
        address = context.xpath('div/div[@id="object-address"]/div/text()').extract()[0]
        item['address']=address
        for a in ADDR_RE:
            if re.match(a, address):
                addr = re.match(a, address).groupdict()
        for key in addr.keys():
            item[key.encode()]=addr[key]
        #square
        try:
            squares = context.xpath('div/div[@id="object-squares"]')
            squares_label = squares.xpath('div[@class="label"]/text()').extract()[0]
            squares_value = squares.xpath('div[@class="value"]/text()').extract()[0]
            if squares_label == u"Площадь (общая):":
                s = re.match(ur"(?P<square>\d+[.]*\d*)[ кв.м.]*", squares_value).groupdict()
            if squares_label == u"Площадь (общая/жилая/кухни):":
                s = re.match(ur"(?P<square>\d+[.]*\d*)[ кв.м.]*/(?P<live_square>\d+[.]*\d*)/(?P<kitchen_square>\d+[.]*\d*)", squares_value).groupdict()
            if squares_label == u"Площадь (общая/жилая):":
                s = re.match(ur"(?P<square>\d+[.]*\d*)[ кв.м.]*/(?P<live_square>\d+[.]*\d*)", squares_value).groupdict()
            if squares_label == u"Площадь (общая/кухни):":
                s = re.match(ur"(?P<square>\d+[.]*\d*)[ кв.м.]*/(?P<kitchen_square>\d+[.]*\d*)", squares_value).groupdict()
            if squares_label == u"Площадь (жилая/кухни):":
                s = re.match(ur"(?P<live_square>\d+[.]*\d*)[ кв.м.]*/(?P<kitchen_square>\d+[.]*\d*)", squares_value).groupdict()
            for key in s.keys():
                item[key.encode()]=s[key]
        except:
            pass
        #rooms
        try:
            rooms = context.xpath('div/div[@id="object-rooms"]/div[@class="value"]/text()').extract()[0].split('/')
            item['rooms'] = rooms[0]
            item['rooms_arrangement'] = rooms[1]
        except:
            pass
        #yes or no fields
        for i in ['bldType', 'furniture', 'metro', 'parking', 'telephone', 'refrigerator', 'tvset']:
            try:
                item[i] = context.xpath('div/div[@id="object-%s"]/div[@class="value"]/text()'%i).extract()[0]
            except:
                pass
        #metro
        try:
            item['metro'] = item['metro'].split(', ')
        except:
            pass
        #floor and floor_count
        try:
            floors = context.xpath('div/div[@id="object-floors"]')
            floors_label = floors.xpath('div[@class="label"]/text()').extract()[0]
            floors_value = floors.xpath('div[@class="value"]/text()').extract()[0].split('/')
            print floors_label, floors_value
            if floors_label == u"Этаж:":
                print floors_label, floors_value
                item['floor'] = floors_value[0]
            if floors_label == u"Этажность:":
                print floors_label, floors_value
                item['floor_count'] = floors_value[0]
            if floors_label == u"Этаж/Этажность:":
                print floors_label, floors_value
                item['floor'] = floors_value[0]
                item['floor_count'] = floors_value[1]
        except:
            pass
        #materials
        try:
            materials = context.xpath('div/div[@id="object-materials"]')
            materials_label = materials.xpath('div[@class="label"]/text()').extract()[0]
            materials_value = materials.xpath('div[@class="value"]/text()').extract()[0]
            if materials_label == u"Материал стен:":
                item['material'] = materials_value
                item['wall_material'] = materials_value
            if materials_label == u"Материал пола:":
                item['floor_material'] = materials_value
            if materials_label == u"Материал перекрытий:":
                item['x_material'] = materials_value
        except:
            pass
        #wc
        try:
            wc = context.xpath('div/div[@id="object-wc"]')
            wc_label = wc.xpath('div[@class="label"]/text()').extract()[0]
            wc_value = wc.xpath('div[@class="value"]/text()').extract()[0].split(',')
            if wc_label == u"Количество, тип санузлов:":
                item['wc_count'] = wc_value[0]
                item['wc_type'] = wc_value[1]
            if wc_label == u"Тип санузла:":
                item['wc_type'] = wc_value[0]
            if wc_label == u"Балкон:":
                item['balcon'] = wc_value[0]
        except:
            pass
        #levels
        try:
            levels = context.xpath('div/div[@id="object-levels"]')
            levels_label = levels.xpath('div[@class="label"]/text()').extract()[0]
            levels_value = levels.xpath('div[@class="value"]/text()').extract()[0]
            if levels_label == u"Ремонт (состояние):":
                item['state'] = levels_value
            if levels_label == u"Количество уровней:":
                item['levels'] = levels_value
        except:
            pass
        #coords
        try:
            coordinates = context.xpath('div/div/div[@id="realty-object-map-data"]')
            item['lat'] = coordinates.xpath('span[@class="lat"]/text()').extract()[0].encode()
            item['lng'] = coordinates.xpath('span[@class="lng"]/text()').extract()[0].encode()
        except:
            pass
        item['code'] = context.xpath('div/div/div[@id="object-code"]/div[@class="value"]/text()').extract()[0]
        #date
        item['update_date'] = context.xpath('div/div/div[@id="object-update-date"]/div[@class="value"]/text()').extract()[0]
        yield item
