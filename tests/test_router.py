"""Tests for cc-router — the part that runs at the consumer.

Its only purpose is a separation: requests for the local model go to the
foreign server, everything else unchanged to Anthropic. If something goes
wrong there, the damage happens at the consumer and nobody notices — which is
exactly why there is more here than the size of the file suggests.
"""
import json, os, unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

import common

R = common.load("setup/claude/cc-router.py", "cc_router")


class RouterOnTheWire(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.at_llama, self.at_anthropic = [], []

        def recorder(liste):
            async def h(request):
                liste.append({"method": request.method,
                              "path": request.path_qs,
                              "headers": {k.lower(): v for k, v in request.headers.items()},
                              "payload": await request.read()})
                return web.json_response({"ok": True})
            return h

        lapp = web.Application(); lapp.router.add_route("*", "/{t:.*}", recorder(self.at_llama))
        aapp = web.Application(); aapp.router.add_route("*", "/{t:.*}", recorder(self.at_anthropic))
        self.lserver, self.aserver = TestServer(lapp), TestServer(aapp)
        await self.lserver.start_server()
        await self.aserver.start_server()

        self.backup = (R.LLAMA, R.UPSTREAM, os.environ.get("LLAMA_API_KEY"))
        R.LLAMA = str(self.lserver.make_url("")).rstrip("/")
        R.UPSTREAM = str(self.aserver.make_url("")).rstrip("/")
        os.environ["LLAMA_API_KEY"] = "geheim-lokal"

        self.server = TestServer(R.build_app())
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")
        self.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        for s in (self.server, self.lserver, self.aserver):
            await s.close()
        R.LLAMA, R.UPSTREAM, key = self.backup
        if key is None:
            os.environ.pop("LLAMA_API_KEY", None)
        else:
            os.environ["LLAMA_API_KEY"] = key

    HEADERS = {
        "authorization": "Bearer sk-ant-oat01-ABO-TOKEN",
        "x-api-key": "sk-ant-api03-SCHLUESSEL",
        "anthropic-auth-token": "sk-ant-oat01-NOCHMAL",
        "anthropic-beta": "oauth-2025-04-20",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "cookie": "session=geheim",
    }

    async def send(self, payload, path="/v1/messages"):
        return await self.session.post(self.url + path, headers=self.HEADERS,
                                       data=json.dumps(payload))


class TestLocalBranch(RouterOnTheWire):
    async def test_goes_to_llama_and_not_to_anthropic(self):
        await self.send({"model": "local/lokal", "messages": []})
        self.assertEqual(len(self.at_llama), 1)
        self.assertEqual(self.at_anthropic, [])

    async def test_prefix_is_stripped(self):
        await self.send({"model": "local/lokal", "messages": []})
        self.assertEqual(json.loads(self.at_llama[0]["payload"])["model"], "lokal")

    async def test_fields_llama_does_not_know_are_dropped(self):
        await self.send({"model": "local/lokal", "messages": [],
                            "thinking": {"type": "enabled"},
                            "context_management": {}, "output_config": {}})
        gesendet = json.loads(self.at_llama[0]["payload"])
        for k in ("thinking", "context_management", "output_config"):
            self.assertNotIn(k, gesendet)

    async def test_no_credentials_reach_the_foreign_server(self):
        """The heart of the matter.

        Previously only authorization and x-api-key were stripped. An
        anthropic-auth-token — which cc-gateway itself knows as a possible
        carrier of the subscription token — would have gone through. The allow
        list cannot repeat that mistake: whatever is new drops out.
        """
        await self.send({"model": "local/lokal", "messages": []})
        headers = self.at_llama[0]["headers"]
        for k in ("anthropic-auth-token", "x-api-key", "cookie", "anthropic-beta"):
            self.assertNotIn(k, headers, "%s went to the foreign server" % k)
        self.assertEqual(headers.get("authorization"), "Bearer geheim-lokal",
                         "only our own access may be set")

    async def test_without_llama_api_key_no_authorization(self):
        os.environ.pop("LLAMA_API_KEY", None)
        await self.send({"model": "local/lokal", "messages": []})
        self.assertNotIn("authorization", self.at_llama[0]["headers"])

    async def test_allowed_headers_arrive(self):
        await self.send({"model": "local/lokal", "messages": []})
        headers = self.at_llama[0]["headers"]
        self.assertEqual(headers.get("anthropic-version"), "2023-06-01")
        self.assertTrue(headers.get("content-type", "").startswith("application/json"))


class TestAnthropicBranch(RouterOnTheWire):
    async def test_foreign_model_goes_to_anthropic_unchanged(self):
        payload = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}],
                 "thinking": {"type": "enabled"}}
        await self.send(payload)
        self.assertEqual(len(self.at_anthropic), 1)
        self.assertEqual(self.at_llama, [])
        self.assertEqual(json.loads(self.at_anthropic[0]["payload"]), payload,
                         "Anthropic must receive byte for byte what the client sent")

    async def test_credential_headers_survive_for_anthropic(self):
        # anthropic-beta carries the OAuth capability marker; without it, 401.
        await self.send({"model": "claude-opus-5", "messages": []})
        headers = self.at_anthropic[0]["headers"]
        self.assertEqual(headers.get("anthropic-beta"), "oauth-2025-04-20")
        self.assertEqual(headers.get("authorization"), "Bearer sk-ant-oat01-ABO-TOKEN")

    async def test_other_paths_go_to_anthropic(self):
        await self.send({"x": 1}, path="/v1/models")
        self.assertEqual(len(self.at_anthropic), 1)
        self.assertEqual(self.at_anthropic[0]["path"], "/v1/models")

    async def test_an_unreadable_body_is_not_rerouted(self):
        r = await self.session.post(self.url + "/v1/messages", headers=self.HEADERS,
                                    data=b"kein json")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.at_llama, [])
        self.assertEqual(self.at_anthropic[0]["payload"], b"kein json")

    async def test_a_model_without_the_prefix_stays_with_anthropic(self):
        await self.send({"model": "lokal", "messages": []})
        self.assertEqual(self.at_llama, [])
        self.assertEqual(len(self.at_anthropic), 1)


if __name__ == "__main__":
    unittest.main()
