# -*- coding: utf-8 -*-
import scrapy
from site_position.items import SitePositionItem

class go_medSpider(scrapy.Spider):
    name = "go_med"
    allowed_domains = ["ru.bookimed.com"]
    start_urls = (
		#'http://ru.bookimed.com/clinics/illness=melanoma/',
		#'http://ru.bookimed.com/clinics/illness=rak-grudi/',
		#'http://ru.bookimed.com/clinics/illness=rak-prostatu/',
		#'http://ru.bookimed.com/clinics/illness=rak-shejki-matki/',
		#'http://ru.bookimed.com/clinics/illness=rak-schitovidnoj-zhelezu/',
		#'http://ru.bookimed.com/clinics/illness=zamena-kolennogo-sustava/',
		#'http://ru.bookimed.com/clinics/illness=zamena-plechevogo-sustava/',
		#'http://ru.bookimed.com/clinics/illness=zamena-tazo-bedrennogo-sustava/',
		#'http://ru.bookimed.com/clinics/illness=rak-podzheludochnoj/',
		#'http://ru.bookimed.com/clinics/illness=epilepsiya',
	    	)

    def parse(self, response):
        total = response.xpath('//span[@id="clinics_list"]/div')
        clinic_name = total[0].xpath('//div[@class="title clinic-title"]/div/a/text()').extract()
        for c in clinic_name:
            item = SitePositionItem()
            item['name'] = c
            yield item
        url = response.urljoin(response.xpath('//ul[@class="pagination"]/li[@class="next"]/a/@href').extract()[0])
        yield scrapy.Request(url, callback=self.parse, dont_filter=True)

