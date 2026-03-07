# How is it checked? Either/or:
    # - by native speaker
    # - from https://www.omniglot.com/language/phrases/hovercraft.htm
    # - the rest is from ChatGPT, so it might be not correct.

hovercraft_translations = {
    "bg": "Моят ховеркрат е пълен с змиорки.",
    "hr": "Moj zračni čamac je pun jegulja.",
    # native speaker checked:
    "cs": "Moje vznášedlo je plné úhořů.", 
    "da": "Mit luftkøretøj er fuld af ål.",
    # omniglot:
    "da": "Mit luftpudefartøj er fyldt med ål.", 
    "nl": "Mijn luchtkussenboot zit vol met palingen.",
    "en": "My hovercraft is full of eels.",
    "et": "Minu õhulaev on täis angerjaid.",
    "fi": "Ilmatyynyalukseni on täynnä ankeriaita.",
    "fr": "Mon aéroglisseur est plein d’anguilles.",
    # omniglot:
    "de": "Mein Luftkissenfahrzeug ist voller Aale.",
    # chatGPT provided, it's wrong:
#        "de": "Mein Luftkissenboot ist voller Aale.",
    "el": "Η αερόστρωσή μου είναι γεμάτη χέλια.",
    "hu": "A légpárnás hajóm tele van angolnákkal.",
    "ga": "Tá mo hovercraft lán le éisc eala.",
    "it": "Il mio hovercraft è pieno di anguille.",
    "lv": "Mans gaisa kuģis ir pilns ar zušiem.",
    "lt": "Mano oro pagalvės laivas pilnas ungurių.",
    "mt": "Il-hovercraft tiegħi huwa mimli b’anguille.",
    # omniglot and chatGPT agree!
    "pl": "Mój poduszkowiec jest pełen węgorzy.",
    # omniglot:
    "pt": "O meu hovercraft está cheio de enguias.",
    # chatGPT provided:
#        "pt": "O meu aerobarco está cheio de enguias.",
    "ro": "Hovercraft-ul meu este plin cu anghile.",
    # omniglot:
    "sk": "Moje vznášadlo je plné úhorov.",
    # chatGPT provided:
#        "sk": "Môj vznášadlo je plné úhorov.",
    "sl": "Moj zračni čoln je poln jegulj.",
    # omniglot and chatGPT agree!
    "es": "Mi aerodeslizador está lleno de anguilas.",
    "sv": "Min luftkuddefarkost är full av ålar.",
    "ar": "المركبة الهوائية الخاصة بي مليئة بالثعابين.",
    "ca": "El meu aerodeslizador està ple d’anguiles.",
    "zh": "我的气垫船满是鳗鱼。",
    "gl": "O meu aerodeslizador está cheo de enguias.",
    "hi": "मेरा होवरक्राफ्ट ईल से भरा हुआ है।",
    "ja": "私のホバークラフトはウナギでいっぱいです。",
    "ko": "내 호버크래프트는 장어로 가득 차 있어요.",
    "no": "Min luftputebåt er full av ål.",
    # omniglot and chatGPT agree!
    "ru": "Моё судно на воздушной подушке полно угрей.",
    "tr": "Hoverkraftım yılanbalıklarıyla dolu.",
    # omniglot, one of versions:
    "uk": "Моє судно на повітряній подушці наповнене вуграми.",
    # chatGPT provided:
#        "uk": "Мій ховеркрафт повний вугрів."
}
def hovercraft_sentence(lang_code):
    return hovercraft_translations[lang_code]
