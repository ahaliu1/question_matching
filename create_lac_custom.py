"""
创建lac_custom.txt
"""

xingzuo = [
    "白羊座", "牡羊座",
    "金牛座",
    "双子座",
    "巨蟹座",
    "狮子座",
    "处女座", "室女座",
    "天秤座", "天平座",
    "天蝎座",
    "射手座", "人马座",
    "摩羯座", "山羊座",
    "水瓶座", "宝瓶座",
    "双鱼座",
]

shengxiao = [
    "鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪",
]

nan_nv = ["男", "女"]

with open("lac_custom.txt", 'w', encoding='utf-8') as f:
    # 星座
    for xz in xingzuo:
        for nn in nan_nv:
            f.write(xz + nn + '\n')
            f.write(nn + xz + '\n')
    # 生肖
    for sx in shengxiao:
        for nn in nan_nv:
            f.write(sx + nn + '\n')
            f.write(nn + sx + '\n')