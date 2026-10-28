"""Local Chinese labels to English YOLO-World prompts; no cloud translation."""
import re

OBJECTS = {
    '纸杯': ['paper cup', 'disposable paper cup'],
    '一次性纸杯': ['paper cup', 'disposable paper cup'],
    '杯子': ['cup', 'mug'], '水杯': ['cup', 'drinking cup'], '马克杯': ['mug'],
    '玻璃杯': ['drinking glass'], '塑料杯': ['plastic cup'], '保温杯': ['travel mug', 'thermos'],
    '水瓶': ['water bottle'], '矿泉水': ['water bottle'], '矿泉水瓶': ['water bottle'],
    '瓶子': ['bottle'], '饮料': ['beverage bottle', 'drink can'], '可乐': ['soda can', 'cola bottle'],
    '易拉罐': ['drink can'], '水壶': ['kettle'], '茶壶': ['teapot'],
    '碗': ['bowl'], '盘子': ['plate'], '勺子': ['spoon'], '叉子': ['fork'], '筷子': ['chopsticks'],
    '水果': ['fruit'], '苹果': ['apple'], '香蕉': ['banana'], '橘子': ['orange'], '橙子': ['orange'],
    '背包': ['backpack'], '书包': ['backpack'], '手提包': ['handbag'], '袋子': ['bag'],
    '手机': ['mobile phone'], '电脑': ['laptop', 'computer monitor'], '笔记本电脑': ['laptop'],
    '键盘': ['computer keyboard'], '鼠标': ['computer mouse'], '遥控器': ['remote control'],
    '耳机': ['headphones'], '充电器': ['charger'], '电线': ['cable'], '插线板': ['power strip'],
    '书': ['book'], '书本': ['book'], '笔记本': ['notebook'], '纸': ['sheet of paper'],
    '笔': ['pen'], '铅笔': ['pencil'], '剪刀': ['scissors'], '胶带': ['adhesive tape'],
    '纸巾': ['tissue box', 'tissue paper'], '纸巾盒': ['tissue box'], '纸盒': ['cardboard box'],
    '盒子': ['box'], '快递盒': ['cardboard box'], '垃圾桶': ['trash can'],
    '椅子': ['chair'], '桌子': ['table'], '沙发': ['sofa'], '床': ['bed'],
    '枕头': ['pillow'], '被子': ['blanket'], '衣服': ['clothes'], '鞋子': ['shoe'],
    '拖鞋': ['slipper'], '帽子': ['hat'], '毛巾': ['towel'], '雨伞': ['umbrella'],
    '钥匙': ['key'], '钱包': ['wallet'], '眼镜': ['eyeglasses'], '时钟': ['clock'],
    '花盆': ['plant pot'], '盆栽': ['potted plant'], '花瓶': ['vase'], '灯': ['lamp'],
    '台灯': ['desk lamp'], '电视': ['television'], '电风扇': ['electric fan'],
    '玩具': ['toy'], '玩偶': ['stuffed toy'], '球': ['ball'], '足球': ['soccer ball'],
    '篮球': ['basketball'], '灭火器': ['fire extinguisher'], '扫把': ['broom'],
}
COLORS = {'红色': 'red', '蓝色': 'blue', '绿色': 'green', '黄色': 'yellow',
          '白色': 'white', '黑色': 'black', '橙色': 'orange', '粉色': 'pink', '紫色': 'purple'}


def resolve_prompts(name, english_hint=''):
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 32:
        raise ValueError('请输入 1–32 字的物品名称')
    name = name.strip()
    if any(ord(c) < 32 for c in name):
        raise ValueError('物品名称不能包含控制字符')
    if not isinstance(english_hint, str) or len(english_hint) > 96:
        raise ValueError('英文识别提示过长')
    english_hint = english_hint.strip()
    if english_hint:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/-]{0,95}", english_hint):
            raise ValueError('英文识别提示请使用英文描述，例如 red paper cup')
        positive = [english_hint.lower()]
    elif re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/-]{0,31}", name):
        positive = [name.lower()]
    else:
        key = name
        color = ''
        for prefix, word in COLORS.items():
            if key.startswith(prefix):
                color, key = word+' ', key[len(prefix):].lstrip('的')
                break
        positive = OBJECTS.get(key)
        if not positive:
            raise ValueError('该中文名称暂无本地译名，请补充英文识别提示；例如 paper cup')
        positive = [color+word for word in positive]
    # Competing classes help avoid labelling every region as the single target.
    negatives = ['person', 'chair', 'table', 'water bottle', 'bowl', 'cardboard box', 'trash can']
    texts = list(dict.fromkeys(positive))
    positive_count = len(texts)
    texts += [word for word in negatives if word not in texts]
    return {'name': name, 'texts': texts, 'positive_count': positive_count}
