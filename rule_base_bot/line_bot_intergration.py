from flask import Flask, request, abort
from flask_pymongo import PyMongo
import pprint
import numpy as np
import requests


from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

app.config["MONGO_URI"] = "mongodb://localhost:27017/pizzabot"
mongo = PyMongo(app)

CONST_FALLBACK_MSG = "무슨 말씀인지 모르겠습니다."
CONST_END_ORDER_MSG = "주문을 종료합니다."

## 본인의 line 봇 정보 입력
line_bot_api = LineBotApi('본인의 line 봇 정보 입력')
handler = WebhookHandler('본인의 line 봇 정보 입력')

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    pprint.pprint(event)
    msg = event.message.text
    user_id = 1

    # user_id 정보로 현재 활성화 되어 있는 대화 인스턴스를 조회
    active_dialogue = get_active_dialogue(user_id)

    # 인텐트 추론
    intent = infer_intent(msg)

    is_no_intent = False
    is_yes_intent = False
    # intent 추론 안되는 경우, yes/np 인텐트의 경우에는 현재 대화 인스턴스의 인텐트를 사용
    if active_dialogue is not None and intent is None:
        intent = active_dialogue['intent']
    elif active_dialogue is not None and intent == 'no':
        intent = active_dialogue['intent']
        is_no_intent = True
    elif active_dialogue is not None and intent == 'yes':
        intent = active_dialogue['intent']
        is_yes_intent = True
    elif intent is None and active_dialogue is None:
        return line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=CONST_FALLBACK_MSG))

    # 인텐트 마스터 데이터 조회
    intent_master = mongo.db.intent.find_one({"name": intent})

    pprint.pprint(intent)
    pprint.pprint(intent_master)
    pprint.pprint(active_dialogue)

    # 메뉴 보여줘와 같은 단순 1회성 인텐트 처리
    if intent_master.get('type') == 'single':
        return process_response(intent_master, active_dialogue, event)

    # no 인텐트의 경우 현재 인텐트에 따라 동작을 분기 처리
    elif is_no_intent is not None and is_no_intent:
        if active_dialogue is None:
            return line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=CONST_FALLBACK_MSG))
        else:
            # 인텐트 마스터 데이터에 등록 되어 있는 no 인텐트에 대한 액션 조회 후 처리
            actions = intent_master['actions']
            if 'no' in actions.keys() and 'skip' == actions['no']:
                change_current_intent(active_dialogue, intent_master)
                return process_response(intent_master, active_dialogue, event)

            elif 'no' in actions.keys() and 'end' == actions['no']:
                end_intent(active_dialogue)
                return line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=CONST_END_ORDER_MSG))

    elif is_yes_intent is not None and is_yes_intent:
        if active_dialogue is None:
            return line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=CONST_FALLBACK_MSG))
        else:
            actions = intent_master['actions']
            if 'yes' in actions.keys() and 'custom' == actions['yes'] and intent == 'order_confirm':
                save_order(active_dialogue, user_id)
                change_current_intent(active_dialogue, intent_master)
                return process_response(intent_master, active_dialogue, event)

    elif 'infer' in intent_master and not intent_master['infer']:
        slots_dialogue = get_slots_from_active_dialogue(active_dialogue)

        slot_name = intent_master['slots'][0]['name']
        slots_dialogue[intent + '.' + slot_name] = msg

        change_current_intent(active_dialogue, intent_master)
        return process_response(intent_master, active_dialogue, event)

    else:
        # userId로 활성화 된 대화 instance가 있는지 조회. 없으면 새로 생성
        active_dialogue = create_new_dialogue_if_not_exist(active_dialogue, intent, user_id)

        # 들어온 문장으로 NER infer 후에 매칭 되는 애들 Slot 전부 저장
        active_dialogue = fill_slot(active_dialogue, intent_master, msg)

        # 마스터데이터로 부터 slot 리스트 갖고와서 대화 instance에 값이 채워져 있는지 확인 for문 돌면서
        # required 인데 안채워져 있으면 새로 채우는 로직 추가
        slots_dialogue = get_slots_from_active_dialogue(active_dialogue)
        if 'slots' in intent_master.keys():
            slots_master = intent_master['slots']
            for slot in slots_master:
                if slot['required'] and intent + '.' + slot['name'] not in slots_dialogue.keys():
                    # required slot 인데 현재 대화 instance 안에 slot filling이 되어 있지 않은 경우 채워야 함
                    return line_bot_api.reply_message(
                            event.reply_token,
                            TextSendMessage(text=slot['response']))

        change_current_intent(active_dialogue, intent_master)
        return process_response(intent_master, active_dialogue, event)


def process_response(intent_master, active_dialogue, event):
    response_msg = intent_master['response']

    if 'response_values' in intent_master.keys():
        slots_dialogue = get_slots_from_active_dialogue(active_dialogue)
        response_values = intent_master['response_values']

        for slot_name in response_values:
            print(slot_name)
            print(slot_name in response_msg)
            if slot_name in response_msg and slot_name in slots_dialogue.keys():
                response_msg = response_msg.replace(slot_name, slots_dialogue[slot_name])
            else:
                response_msg = response_msg.replace(slot_name, '')

    return line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response_msg))


def get_active_dialogue(user_id):
    active_dialogue = list(
        mongo.db.dialogue.find({"user_id": user_id, "status": "active"}).sort([("_id", -1)]).limit(1))

    if active_dialogue is not None and len(active_dialogue) > 0:
        return active_dialogue[0]
    else:
        return None


def create_new_dialogue_if_not_exist(active_dialogue, intent, user_id):
    if active_dialogue is None:
        active_dialogue = save_new_dialogue(intent, user_id)
    return active_dialogue


def save_new_dialogue(intent, user_id):
    new_dialogue = {"user_id": user_id, "status": "active", "intent": intent}
    mongo.db.dialogue.insert_one(new_dialogue)
    active_dialogue = list(
        mongo.db.dialogue.find({"user_id": user_id, "status": "active"}).sort([("_id", -1)]).limit(1))
    return active_dialogue[0]


def get_slots_from_active_dialogue(active_dialogue):
    if 'slots' in active_dialogue.keys():
        slots_dialogue = active_dialogue['slots']
    else:
        slots_dialogue = {}
    return slots_dialogue


def save_order(active_dialogue, user_id):
    slots = active_dialogue['slots']
    order_info = {'user_id': user_id,
                  'menu': slots['order.menu'],
                  'size': slots['order.size'],
                  'address': slots['address.address']}

    if 'side-menu-order.menu' in slots:
        order_info['side-menu'] = slots['side-menu-order.menu']

    mongo.db.order.insert_one(order_info)


def change_current_intent(active_dialogue, intent_master):
    if 'next_intent' not in intent_master:
        end_intent(active_dialogue)
        return

    active_dialogue['intent'] = intent_master['next_intent']
    mongo.db.dialogue.replace_one({"_id": active_dialogue['_id']}, active_dialogue)


def end_intent(active_dialogue):
    active_dialogue['status'] = 'end'
    mongo.db.dialogue.replace_one({"_id": active_dialogue['_id']}, active_dialogue)


def fill_slot(active_dialogue, intent_master, msg):
    entities = infer_ner(msg)

    if entities is None:
        return active_dialogue

    slots_master = intent_master['slots']
    slots_dialogue = get_slots_from_active_dialogue(active_dialogue)

    names_of_slots = [n['name'] for n in slots_master]

    for entity in entities:
        if entity['name'] in names_of_slots:
            slots_dialogue[intent_master['name'] + '.' + entity['name']] = entity['value']

    active_dialogue['slots'] = slots_dialogue
    mongo.db.dialogue.replace_one({"_id": active_dialogue['_id']}, active_dialogue)

    return active_dialogue


def infer_intent(msg):
    result = requests.get('http://localhost:5002/intent?sentence='+msg)

    print('INTENT----------------------------------------------------------')
    print(result.status_code)
    print(result.text)

    if result.status_code == 500:
        return None

    json = result.json()

    return json['result']

def infer_ner(msg):
    result = requests.get('http://localhost:5001/ner?sentence=' + msg)
    json = result.json()
    print('NER----------------------------------------------------------')
    print(json['result'])
    return json['result']

if __name__ == '__main__':

    app.run()
