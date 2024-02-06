# pip install py-hanspell
from hanspell import spell_checker

# 맞춤법 검사 함수
def check_spelling(text):
    result = spell_checker.check(text)
    return result.errors  # 오탈자가 있는 경우 True 반환