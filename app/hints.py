import asyncio
import base64
import json
from typing import List, Optional

import aiohttp
from aiohttp import ClientTimeout

from .config import FB_CONCURRENCY, FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, logger
from .models import QAPair


class FusionBrainClient:
    """Клиент FusionBrain (Kandinsky) API."""

    BASE_URL = "https://api-key.fusionbrain.ai/"

    def __init__(self, api_key: str, secret: str, session: aiohttp.ClientSession):
        self.api_key = api_key
        self.secret = secret
        self.session = session
        self._pipeline_id: Optional[str] = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-Key": f"Key {self.api_key}",
            "X-Secret": f"Secret {self.secret}",
        }

    async def _ensure_pipeline(self) -> Optional[str]:
        if self._pipeline_id:
            return self._pipeline_id
        url = f"{self.BASE_URL}key/api/v1/pipelines"
        async with self.session.get(url, headers=self.headers, timeout=ClientTimeout(60)) as response:
            if response.status >= 400:
                text = await response.text()
                logger.warning(
                    "FusionBrain pipelines responded %s: %s", response.status, text or "empty body"
                )
                return None
            pipelines = await response.json()
        if not pipelines:
            logger.warning("FusionBrain не вернул список pipelines")
            return None
        kandinsky = next(
            (
                pipe
                for pipe in pipelines
                if "kandinsky" in (pipe.get("name", "") + pipe.get("description", "")).lower()
                or pipe.get("type", "").lower() in {"text2image", "text_to_image"}
            ),
            pipelines[0],
        )
        pipeline_id = kandinsky.get("id")
        if not pipeline_id:
            logger.warning("FusionBrain: не удалось определить pipeline_id")
            return None
        self._pipeline_id = pipeline_id
        return pipeline_id

    async def text2image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_images: int = 1,
        style: str = "",
        poll_interval: float = 1.2,
        timeout_s: int = 120,
    ) -> List[bytes]:
        pipeline_id = await self._ensure_pipeline()
        if not pipeline_id:
            return []
        url_run = f"{self.BASE_URL}key/api/v1/pipeline/run"
        generate_params = {"query": prompt}
        if style:
            generate_params["style"] = style
        params = {
            "type": "GENERATE",
            "numImages": num_images,
            "width": width,
            "height": height,
            "generateParams": generate_params,
        }

        form = aiohttp.FormData()
        form.add_field("pipeline_id", str(pipeline_id), content_type="text/plain")
        form.add_field("params", json.dumps(params), content_type="application/json")

        async with self.session.post(url_run, headers=self.headers, data=form, timeout=ClientTimeout(60)) as response:
            if response.status >= 400:
                text = await response.text()
                logger.warning(
                    "FusionBrain pipeline/run responded %s: %s", response.status, text or "empty body"
                )
                return []
            data = await response.json()
        uuid = data.get("uuid")
        if not uuid:
            logger.warning("FusionBrain: не найден uuid в ответе API")
            return []

        url_status = f"{self.BASE_URL}key/api/v1/pipeline/status/{uuid}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        while True:
            async with self.session.get(url_status, headers=self.headers, timeout=ClientTimeout(60)) as response:
                if response.status >= 400:
                    text = await response.text()
                    logger.warning(
                        "FusionBrain pipeline/status responded %s: %s",
                        response.status,
                        text or "empty body",
                    )
                    return []
                status_payload = await response.json()
            status = status_payload.get("status")
            if status == "DONE":
                images_b64 = status_payload.get("images") or []
                if not images_b64:
                    logger.info("FusionBrain вернул DONE без изображений")
                    return []
                return [base64.b64decode(b64) for b64 in images_b64]
            if status in {"FAIL", "ERROR"}:
                logger.warning("FusionBrain вернул статус ошибки: %s", status_payload)
                return []
            if asyncio.get_event_loop().time() > deadline:
                logger.warning("FusionBrain: ожидание генерации превысило лимит времени")
                return []
            await asyncio.sleep(poll_interval)


def build_hint_prompt(question: str, answer: str) -> str:
    return (
        "Создай атмосферную и понятную иллюстрацию по мотивам вопроса викторины. "
        "Она должна намекать на правильный ответ, но не выдавать его напрямую.\n\n"
        f"Вопрос: {question}\n"
        f"Правильный ответ: {answer}\n\n"
        "Избегай текста, водяных знаков, логотипов и букв на изображении."
    )


async def generate_hint_for_pair(
    pair: QAPair,
    fb: FusionBrainClient,
    sem: asyncio.Semaphore,
    timeout: int = 30,
) -> None:
    if pair.hint_image:
        return
    prompt = build_hint_prompt(pair.question, pair.answer)
    logger.info("Генерация подсказки для вопроса: %s", pair.question[:80])
    try:
        async with sem:
            images = await asyncio.wait_for(
                fb.text2image(prompt=prompt, width=768, height=768, num_images=1, timeout_s=timeout),
                timeout=timeout,
            )
        if images:
            pair.hint_image = images[0]
            logger.info("Подсказка получена: %s", pair.question[:80])
        else:
            logger.info("FusionBrain не предоставил подсказку для вопроса: %s", pair.question[:80])
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось получить подсказку для '%s': %s", pair.question[:40], exc)


async def generate_hints_for_pairs(pairs: List[QAPair], timeout: int = 45) -> None:
    if not pairs or not (FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET):
        return
    targets = [pair for pair in pairs if not pair.hint_image]
    if not targets:
        return
    concurrency = max(1, FB_CONCURRENCY)
    try:
        async with aiohttp.ClientSession() as session:
            fb = FusionBrainClient(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, session)
            sem = asyncio.Semaphore(concurrency)
            tasks = [generate_hint_for_pair(pair, fb, sem, timeout=timeout) for pair in targets]
            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось получить подсказки: %s", exc)


async def test_fusionbrain_api() -> None:
    if not (FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET):
        logger.warning("FusionBrain API не настроен — пропускаем проверку")
        return
    try:
        async with aiohttp.ClientSession() as session:
            fb = FusionBrainClient(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, session)
            sem = asyncio.Semaphore(1)
            dummy = QAPair(question="Что такое Искусственный интеллект?", answer="Алгоритмы")
            await generate_hint_for_pair(dummy, fb, sem, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.error("Проверка FusionBrain API завершилась ошибкой: %s", exc)
