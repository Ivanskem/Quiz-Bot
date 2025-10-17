from __future__ import annotations

import asyncio

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from .config import FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, logger
from .hints import generate_hints_for_pairs
from .models import UserSession, get_session
from .qa import generate_qa_pairs
from .recommendations import RecommendationContext, build_recommendation
from .text_utils import is_correct
from .tts import text_to_speech_gtts

router = Router()


@router.message(CommandStart())
async def start(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    session = get_session(user.id)
    session.__dict__.update(UserSession().__dict__)
    await message.answer(
        "Привет! Это *Квиз-бот*, который сможет подготовить вопросы по твоему тексту.",
        parse_mode="Markdown",
    )
    await message.answer(
        "Пришли мне текст (минимум 200 символов), после чего выбери дальнейшее действие.",
    )
    session.stage = "IDLE"


@router.message(F.text.func(lambda text: text and len(text) >= 200))
async def receive_text(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    session = get_session(user.id)
    session.text = message.text or ""
    session.stage = "CHOOSE_ACTION"

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="Озвучить текст", callback_data="tts")
    keyboard.button(text="Создать викторину", callback_data="quiz")
    keyboard.adjust(1)

    await message.answer(
        "Готово! Выберите, что сделать дальше:", reply_markup=keyboard.as_markup()
    )


@router.callback_query(F.data == "tts")
async def on_tts(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    if session.stage != "CHOOSE_ACTION":
        await callback.answer()
        return
    await callback.answer("Готовим озвучку…")
    if not callback.message:
        return
    try:
        audio_data = await asyncio.to_thread(text_to_speech_gtts, session.text)
        await callback.message.answer_audio(
            audio=BufferedInputFile(audio_data, filename="voice.mp3"),
            caption="Ваш текст в аудиоформате.",
        )
    except Exception as exc:  # noqa: BLE001
        await callback.message.answer(
            f"Не получилось озвучить текст: {exc}. Проверьте работу gTTS."
        )
        logger.exception("Ошибка gTTS")
        return

    await callback.message.answer("Теперь можно создать вопросы, если это необходимо.")


@router.callback_query(F.data == "quiz")
async def on_quiz(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    if session.stage != "CHOOSE_ACTION":
        await callback.answer()
        return
    await callback.answer("Начинаем подготовку квиза…")
    if not callback.message:
        return
    session.stage = "WAIT_NUM"
    await callback.message.answer("Сколько вопросов нужно сгенерировать? (1-20)")


@router.message(F.text.regexp(r"^\d{1,2}$"))
async def receive_num(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    session = get_session(user.id)
    if session.stage != "WAIT_NUM":
        return
    count = int(message.text or 0)
    n_questions = max(1, min(20, count))
    session.num_questions = n_questions
    await message.answer("Формирую вопросы…")

    try:
        session.qa = await generate_qa_pairs(session.text, n_questions)
        await generate_hints_for_pairs(session.qa, timeout=35)
        session.idx = session.correct = session.wrong = 0
        session.stage = "ASKING"
        await ask_next_question(message, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Не удалось подготовить викторину")
        await message.answer(f"Возникла ошибка при генерации вопросов: {exc}")
        session.stage = "IDLE"


async def ask_next_question(message: Message, session: UserSession) -> None:
    if session.idx >= len(session.qa):
        await send_stats(message, session)
        return

    qa = session.qa[session.idx]
    if not qa.hint_image:
        try:
            await generate_hints_for_pairs([qa], timeout=30)
        except Exception as exc:  # noqa: BLE001
            logger.error("Подсказка для вопроса %s не получена: %s", session.idx + 1, exc)

    keyboard = InlineKeyboardBuilder()
    keyboard.button(text="Попробовать ещё раз", callback_data="retry")
    keyboard.button(text="Показать ответ", callback_data="reveal")
    keyboard.adjust(1, 1)

    caption = f"Вопрос {session.idx + 1}/{len(session.qa)}:\n\n{qa.question}"

    if qa.hint_image:
        photo = BufferedInputFile(qa.hint_image, filename="hint.jpg")
        await message.answer_photo(photo=photo, caption=caption, reply_markup=keyboard.as_markup())
    else:
        extra = (
            "\n\n(Подсказка пока не доступна — возможно, сервис генерации изображений временно недоступен.)"
            if FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET
            else ""
        )
        await message.answer(caption + extra, reply_markup=keyboard.as_markup())

    session.stage = "WAIT_ANSWER"


@router.message(F.text)
async def receive_answer(message: Message) -> None:
    user = message.from_user
    if not user:
        return
    session = get_session(user.id)
    if session.stage != "WAIT_ANSWER" or session.idx >= len(session.qa):
        if message.text and len(message.text) >= 200:
            await receive_text(message)
        return
    if not message.text:
        return

    qa = session.qa[session.idx]
    if is_correct(message.text, qa.answer):
        session.correct += 1
        caption = f"Верно! Правильный ответ: *{qa.answer}*"
        if qa.hint_image:
            photo = BufferedInputFile(qa.hint_image, filename="hint.jpg")
            await message.answer_photo(photo=photo, caption=caption, parse_mode="Markdown")
        else:
            await message.answer(caption, parse_mode="Markdown")
        session.idx += 1
        session.stage = "ASKING"
        await ask_next_question(message, session)
    else:
        await message.answer(
            "Не совсем так. Попробуйте ещё раз или нажмите кнопку, чтобы увидеть правильный ответ."
        )


@router.callback_query(F.data == "retry")
async def on_retry(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    if session.stage != "WAIT_ANSWER":
        await callback.answer()
        return
    await callback.answer("Пробуем ещё раз!")
    if callback.message:
        await callback.message.answer("Попробуйте сформулировать ответ иначе — подсказка по-прежнему доступна.")


@router.callback_query(F.data == "reveal")
async def on_reveal(callback: CallbackQuery) -> None:
    session = get_session(callback.from_user.id)
    if session.stage != "WAIT_ANSWER":
        await callback.answer()
        return
    qa = session.qa[session.idx]
    message_obj = callback.message
    if not isinstance(message_obj, Message):
        return
    session.wrong += 1
    await callback.answer()

    caption = f"Правильный ответ: *{qa.answer}*"
    if qa.hint_image:
        photo = BufferedInputFile(qa.hint_image, filename="hint.jpg")
        await message_obj.answer_photo(photo=photo, caption=caption, parse_mode="Markdown")
    else:
        await message_obj.answer(caption, parse_mode="Markdown")

    session.idx += 1
    session.stage = "ASKING"
    await ask_next_question(message_obj, session)


async def send_stats(message: Message, session: UserSession) -> None:
    total = len(session.qa)
    stats_text = (
        "Раунд завершён!\n\n"
        f"Верных ответов: {session.correct}\n"
        f"Ошибок/пропусков: {session.wrong}\n"
        f"Всего вопросов: {total}"
    )
    recommendation = build_recommendation(
        RecommendationContext(
            total=total,
            correct=session.correct,
            wrong=session.wrong,
            text_chars=len(session.text),
        )
    )
    await message.answer(f"{stats_text}\n\nСовет эксперта: {recommendation}")

    session.stage = "IDLE"
    session.text = ""
    session.qa.clear()
    session.idx = session.correct = session.wrong = 0
    session.num_questions = 0
