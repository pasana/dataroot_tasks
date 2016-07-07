# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class Spider1000RealtyItem(scrapy.Item):
    type = scrapy.Field()
    price = scrapy.Field()
    description = scrapy.Field()
    street = scrapy.Field() 
    house_number = scrapy.Field()
    area = scrapy.Field()
    district = scrapy.Field()
    city = scrapy.Field()
    rooms = scrapy.Field()
    rooms_arrangement = scrapy.Field()
    square = scrapy.Field()
    kitchen_square = scrapy.Field()
    live_square = scrapy.Field()
    metro = scrapy.Field()
    parking = scrapy.Field()
    wc_count = scrapy.Field()
    wc_type = scrapy.Field()
    balcon = scrapy.Field()
    telephone = scrapy.Field()
    refrigerator = scrapy.Field()
    tvset = scrapy.Field()
    material = scrapy.Field()
    floor_material = scrapy.Field()
    wall_material = scrapy.Field()
    x_material = scrapy.Field()
    floor = scrapy.Field()
    floor_count = scrapy.Field()
    lat = scrapy.Field()
    lng = scrapy.Field()
    code = scrapy.Field()
    update_date = scrapy.Field()
    levels = scrapy.Field()
    state = scrapy.Field()
    address = scrapy.Field()
