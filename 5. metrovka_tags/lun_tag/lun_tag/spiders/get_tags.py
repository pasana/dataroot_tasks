# -*- coding: utf-8 -*-
import scrapy
from lun_tag.items import LunTagItem

class GetTagsSpider(scrapy.Spider):
    name = "get_tags"
    allowed_domains = ["lun.ua"]
    start_urls = (
        'http://www.lun.ua/%D0%B0%D1%80%D0%B5%D0%BD%D0%B4%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80-%D0%BA%D0%B8%D0%B5%D0%B2',
        'http://www.lun.ua/%D0%B0%D1%80%D0%B5%D0%BD%D0%B4%D0%B0-%D0%BA%D0%B2%D0%B0%D1%80%D1%82%D0%B8%D1%80-%D0%BA%D0%B8%D0%B5%D0%B2?page=2'
    )

    def parse(self, response):
        results = response.xpath('//div[@class="obj"]')
        print "LEN", len(results)
        for r in results:
            item = LunTagItem()
            try:
                item['url'] = 'lun.ua' + r.xpath('div/div/h3/span/noindex/a/@href').extract()[0]
                item['tags']= r.xpath('div[@class="wrap mt-10"]/text()').extract()[1].strip()
                print item
                yield item
            except:
                pass
        try:
            page_num = response.xpath('//div[@class="pagination pagination_center"]/a[6]/@data-value').extract()[0]
            print page_num
            url = response.urljoin("?page=%s"%page_num)
            yield scrapy.Request(url, callback=self.parse, dont_filter=True)
        except:
            pass
