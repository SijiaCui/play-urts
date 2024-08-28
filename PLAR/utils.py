
def en2zh(query: str, from_lang='en', to_lang='zh') -> str:
    import requests
    import random
    import json
    from hashlib import md5

    # Set your own appid/appkey.
    appid = '20220906001333452'
    appkey = 'Ndqv6OJANV0UuU1elZpe'

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    # from_lang = 'en'
    # to_lang =  'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # json.dumps(result, indent=4, ensure_ascii=False)
    result = result['trans_result'][0]['dst']

    return result


def test_en2zh():
    # https://ancient-warriors.fandom.com/wiki/Ranged_Soldier
    sentence = """
A Ranged Soldier is troop that fires projectiles from a distance to eliminate enemy soldiers without putting themselves in danger. These soldiers specialised in slowly weakening Heavy Infantry and killing swarms of Light Infantry with a mass volley of arrows. They lacked in melee combat having to use simple weapons like short swords or daggers they keep on their side. Ranged Soldiers usually lake in armour but make up for it in mobility and can be used effectively alongside Light Soldiers.
"""
    sentence_zh = en2zh(sentence)
    print(sentence_zh)


if __name__ == "__main__":
    test_en2zh()
    pass
