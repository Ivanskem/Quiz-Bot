import asyncio
import base64
import json
from typing import Any, List, Optional

import aiohttp
from aiohttp import ClientTimeout

from .config import FB_CONCURRENCY, FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, logger
from .models import QAPair


def _looks_like_b64(s: str) -> bool:
    if not s or len(s) < 64:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    if any(ch not in allowed for ch in s[:256]):
        return False
    return (len(s.replace("\n", "").replace("\r", "")) % 4) == 0


def _extract_b64_candidates(obj: Any, limit: int = 4) -> List[str]:
    out: List[str] = []
    def walk(x: Any) -> None:
        nonlocal out
        if len(out) >= limit:
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
                if len(out) >= limit:
                    return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if len(out) >= limit:
                    return
        elif isinstance(x, str):
            if _looks_like_b64(x):
                out.append(x)
    walk(obj)
    return out


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
            "Accept": "application/json",
        }

    async def _ensure_pipeline(self) -> Optional[str]:
        if self._pipeline_id:
            return self._pipeline_id
        url = f"{self.BASE_URL}key/api/v1/pipelines"
        async with self.session.get(url, headers=self.headers, timeout=ClientTimeout(60)) as r:
            txt = await r.text()
            if r.status >= 400:
                logger.warning("[FB] pipelines %s: %s", r.status, (txt or "<empty>")[:400])
                return None
            try:
                pipelines = json.loads(txt)
            except Exception:
                logger.warning("[FB] pipelines parse error: %s", (txt or "<empty>")[:400])
                return None
        if not pipelines:
            logger.warning("[FB] pipelines: empty list")
            return None
        picked = next(
            (p for p in pipelines if isinstance(p, dict) and (
                "kandinsky" in (p.get("name", "") + p.get("description", "")).lower()
                or str(p.get("type", "")).lower() in {"text2image", "text_to_image"}
            )),
            pipelines[0],
        )
        pid = picked.get("id") if isinstance(picked, dict) else None
        if not pid:
            logger.warning("[FB] pipeline id missing in: %s", picked)
            return None
        self._pipeline_id = pid
        return pid

    async def text2image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_images: int = 1,
        style: str = "",
        poll_interval: float = 1.0,
        timeout_s: int = 120,
    ) -> List[bytes]:
        pid = await self._ensure_pipeline()
        if not pid:
            return []
        url_run = f"{self.BASE_URL}key/api/v1/pipeline/run"
        gen_params: dict[str, Any] = {"query": prompt}
        if style:
            gen_params["style"] = style
        params: dict[str, Any] = {
            "type": "GENERATE",
            "numImages": num_images,
            "width": width,
            "height": height,
            "generateParams": gen_params,
        }
        form = aiohttp.FormData()
        form.add_field("pipeline_id", str(pid), content_type="text/plain")
        form.add_field("params", json.dumps(params), content_type="application/json")

        async with self.session.post(url_run, headers=self.headers, data=form, timeout=ClientTimeout(60)) as r:
            run_txt = await r.text()
            logger.info("[FB] run %s: %s", r.status, (run_txt or "<empty>")[:300])
            if r.status >= 400:
                return []
            try:
                data = json.loads(run_txt)
            except Exception:
                logger.warning("[FB] run parse error: %s", (run_txt or "<empty>")[:300])
                return []
        uuid = data.get("uuid") if isinstance(data, dict) else None
        if not uuid:
            logger.info("[FB] run: no uuid in %s", data)
            return []

        url_status = f"{self.BASE_URL}key/api/v1/pipeline/status/{uuid}"
        deadline = asyncio.get_event_loop().time() + timeout_s
        poll = 0
        while True:
            async with self.session.get(url_status, headers=self.headers, timeout=ClientTimeout(60)) as r:
                st_txt = await r.text()
                if r.status >= 400:
                    logger.info("[FB] status %s: %s", r.status, (st_txt or "<empty>")[:300])
                    return []
                try:
                    payload = json.loads(st_txt)
                except Exception:
                    logger.info("[FB] status parse error: %s", (st_txt or "<empty>")[:300])
                    return []
            poll += 1
            logger.info("[FB] poll #%s: %s", poll, payload)
            status = payload.get("status") if isinstance(payload, dict) else None
            if status == "DONE":
                images_b64 = payload.get("images") if isinstance(payload, dict) else None
                if images_b64:
                    return [base64.b64decode(b) for b in images_b64]
                candidates = _extract_b64_candidates(payload)
                if candidates:
                    return [base64.b64decode(candidates[0])]
                logger.info("[FB] DONE with no images")
                return []
            if status in {"FAIL", "ERROR"}:
                logger.info("[FB] status error: %s", payload)
                return []
            if asyncio.get_event_loop().time() > deadline:
                logger.info("[FB] status timeout")
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


async def generate_hint_for_pair(pair: QAPair, fb: FusionBrainClient, timeout: int = 30, worker_id: Optional[int] = None) -> None:
    if pair.hint_image:
        return
    prefix = f"[Worker-{worker_id}] " if worker_id is not None else ""
    prompt = build_hint_prompt(pair.question, pair.answer)
    logger.info("%sГенерация подсказки для вопроса: %s", prefix, pair.question[:80])
    try:
        images = await fb.text2image(prompt=prompt, width=768, height=768, num_images=1, timeout_s=timeout)
        if images:
            pair.hint_image = images[0]
            logger.info("%sПодсказка получена", prefix)
        else:
            logger.info("%sПодсказка не получена (пустой ответ)", prefix)
    except Exception as exc:  # noqa: BLE001
        logger.error("%sОшибка генерации подсказки: %s", prefix, exc)


async def generate_hints_for_pairs(pairs: List[QAPair], timeout: int = 45) -> None:
    if not pairs or not (FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET):
        return
    targets = [p for p in pairs if not p.hint_image]
    if not targets:
        return
    concurrency = max(1, FB_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        fb = FusionBrainClient(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, session)
        q: asyncio.Queue[QAPair] = asyncio.Queue()
        for p in targets:
            q.put_nowait(p)

        async def worker(idx: int) -> None:
            while True:
                try:
                    p = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await generate_hint_for_pair(p, fb, timeout=timeout, worker_id=idx)
                finally:
                    q.task_done()

        tasks = [asyncio.create_task(worker(i+1)) for i in range(concurrency)]
        await q.join()
        for t in tasks:
            t.cancel()
        # swallow cancellations
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_fusionbrain_api() -> None:
    if not (FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET):
        logger.warning("FusionBrain API не настроен — пропускаем проверку")
        return
    try:
        async with aiohttp.ClientSession() as session:
            fb = FusionBrainClient(FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET, session)
            dummy = QAPair(question="Что такое Искусственный интеллект?", answer="Алгоритмы")
            await generate_hint_for_pair(dummy, fb, timeout=10, worker_id=0)
    except Exception as exc:  # noqa: BLE001
        logger.error("Проверка FusionBrain API завершилась ошибкой: %s", exc)
