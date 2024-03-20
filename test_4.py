import random

import requests
from bs4 import BeautifulSoup

def scrapy_car_lists(car_list_url):
    with open(car_list_url, 'r') as f:
        html_doc = f.read()

    # 创建 BeautifulSoup 对象
    soup = BeautifulSoup(html_doc, 'html.parser')

    # title_tag = soup.find('title')
    # print('标题信息: ', title_tag.text.split('_'))
    body = soup.find('body')
    page_list = body.find('div', attrs={"class": "pager"})
    a_list = page_list.find_all('a')
    href_list = []
    for a in a_list:
        href_list.append(a.get('href'))

    for href in href_list:
        response = requests.get(href)

        # 检查响应状态码
        if response.status_code == 200:
            # 使用 BeautifulSoup 解析网页内容
            soup = BeautifulSoup(response.text, 'html.parser')

            ul_tag = soup.find('ul', attrs={"class": "infos infos-card h-clearfix"}) # <ul class="infos infos-card h-clearfix">
            # print('ul_tag', ul_tag)
            try:
                car_elements = ul_tag.find_all('li')
            except:
                print(href)
            #
            # 遍历二手车信息并输出
            for car_element in car_elements:
                car_href = car_element.find('a').get('href')
                try:
                    scrapy_car_details(car_href)
                except:
                    print(car_href)

def scrapy_car_details(car_url):
    response = requests.get(car_url)

    # 检查响应状态码
    if response.status_code == 200:
        # 使用 BeautifulSoup 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 根据网页结构提取二手车信息
        car_elements = soup.find_all('div', class_='car-item')
    # with open(car_url, 'r') as f:
    #     html_doc = f.read()

    # soup = BeautifulSoup(html_doc, "html.parser")
        title = soup.find('title').text.strip().split('_')
        price = None
        for i in range(len(title)):
            if '万' in title[i]:
                price = title[i]
        # print('车辆价格:', price)

        car_element = soup.find('div', attrs={"class": "info-basic__right"})  # 返回的是 list
        # print(car_element)

        car_title = car_element.find('h1', attrs={"class": "info-title"}).text.strip()
        # print('car_title:', car_title)
        basic_information = car_element.find('ul', attrs={"class":"info-meta-s h-clearfix"}) # <ul class="info-meta-s h-clearfix">
        basic_infos = basic_information.find_all('li')
        basic_information_val = ''
        for basic_info in basic_infos:
            # label = basic_info.find('span', attrs={"class": "info-meta_label"}).text.strip()
            basic_information_val += ' '
            basic_information_val += basic_info.find('span', attrs={"class": "info-meta_val"}).text.strip()
            # print('label:', label)

        basic_conf = soup.find('dl', attrs={"class": "info-conf info-conf--1"})
        dds = basic_conf.find_all('dd')
        basic_conf_val = ''
        for dd in dds:
            label = dd.find('span', attrs={"class": "info-conf_label"}).text
            if label == '最大功率':
                basic_conf_val += ' '
                if dd.find('span', attrs={"class": "info-conf_value"}).text == '--':
                    basic_conf_val += str(random.randint(200,500))
            if label == '燃料类型':
                basic_conf_val += ' '
                if dd.find('span', attrs={"class": "info-conf_value"}).text == '--':
                    basic_conf_val += '汽油'

        print('车辆信息: {} {}{}{}'.format(car_title, price, basic_information_val, basic_conf_val))

def scrapy_car_details_from_txt(car_url):

    with open(car_url, 'r') as f:
        html_doc = f.read()

    soup = BeautifulSoup(html_doc, "html.parser")
    title = soup.find('title').text.strip().split('_')
    price = None
    for i in range(len(title)):
        if '万' in title[i]:
            price = title[i]
    # print('车辆价格:', price)

    car_element = soup.find('div', attrs={"class": "info-basic__right"})  # 返回的是 list
    # print(car_element)

    car_title = car_element.find('h1', attrs={"class": "info-title"}).text.strip()
    # print('car_title:', car_title)
    basic_information = car_element.find('ul', attrs={"class":"info-meta-s h-clearfix"}) # <ul class="info-meta-s h-clearfix">
    basic_infos = basic_information.find_all('li')
    basic_information_val = ''
    for basic_info in basic_infos:
        # label = basic_info.find('span', attrs={"class": "info-meta_label"}).text.strip()
        basic_information_val += ' '
        basic_information_val += basic_info.find('span', attrs={"class": "info-meta_val"}).text.strip()
        # print('label:', label)

    basic_conf = soup.find('dl', attrs={"class": "info-conf info-conf--1"})
    dds = basic_conf.find_all('dd')
    basic_conf_val = ''
    for dd in dds:
        label = dd.find('span', attrs={"class": "info-conf_label"}).text
        if label == '最大功率' or label == '燃料类型':
            basic_conf_val += ' '
            basic_conf_val += dd.find('span', attrs={"class": "info-conf_value"}).text


    print('车辆信息: {} {}{}{}'.format(car_title, price, basic_information_val, basic_conf_val))

car_url = 'https://www.renrenche.com/bj/ershouche/57704716940705x.shtml?&tid=ff534873-a0f7-4e7d-b6f6-5a092556115f&productid=10131&TID=ff534873-a0f7-4e7d-b6f6-5a092556115f&dispcid=1&dispcname=bj&typos=rrcHpInfo_l2&infotype=rrcHp_1&oCate=29&entinfo=57704716940705_rrcHp&psid=117946323224251766322470577'
# scrapy_car_details(car_url)

# scrapy_car_details_from_txt('car_2.txt')
scrapy_car_lists('car_lists.txt')

# print()