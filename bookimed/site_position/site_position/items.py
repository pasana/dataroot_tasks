# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class SitePositionItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    clinic_id = scrapy.Field()
    name = scrapy.Field()
    date = scrapy.Field()
    full_rate = scrapy.Field()
    rate_num = scrapy.Field()
    good = scrapy.Field()
    bad = scrapy.Field()
    review = scrapy.Field()
    pass
