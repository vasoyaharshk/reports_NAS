"""Offline unit tests for Telegram command parsing and report dispatch."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import telegram_reports as bot


class TelegramReportTests(unittest.TestCase):
    def test_help_lists_configured_reports(self):
        with patch.object(bot.settings, "TELEGRAM_REPORTS", {
            "weekly": {"description": "Weekly comparison"}
        }):
            self.assertIn("/weekly - Weekly comparison", bot.help_text())

    def test_unauthorized_chat_does_not_run_commands(self):
        with patch.object(bot.settings, "TELEGRAM_CHAT_ID", "123"), patch.object(
            bot, "send_message"
        ) as send, patch.object(bot, "generate_report") as generate:
            bot.handle_message({"chat": {"id": 999}, "text": "/weekly"})
            generate.assert_not_called()
            send.assert_not_called()

    def test_shortcut_and_fresh_argument(self):
        entry = {"weekly": {"module": "Weekly_Daily_Trials", "files": ["a.pdf"]}}
        with patch.object(bot.settings, "TELEGRAM_CHAT_ID", "123"), patch.object(
            bot.settings, "TELEGRAM_REPORTS", entry
        ), patch.object(bot, "send_message") as send, patch.object(
            bot, "send_document"
        ) as document, patch.object(
            bot, "generate_report", return_value=[Path("a.pdf")]
        ) as generate:
            bot.handle_message({"chat": {"id": "123"}, "text": "/report weekly fresh"})
            generate.assert_called_once_with("weekly", fresh=True)
            document.assert_called_once_with(Path("a.pdf"))
            self.assertEqual(send.call_count, 2)

    def test_disallowed_unknown_report(self):
        with patch.object(bot.settings, "TELEGRAM_CHAT_ID", "123"), patch.object(
            bot, "send_message"
        ) as send, patch.object(bot, "generate_report") as generate:
            bot.handle_message({"chat": {"id": "123"}, "text": "/not_configured"})
            generate.assert_not_called()
            self.assertIn("Unknown report", send.call_args.args[0])

    def test_output_path_traversal_is_rejected_before_data_load(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            bot.settings, "REPORT_DIRECTORY", Path(folder)
        ), patch.object(bot.settings, "TELEGRAM_REPORTS", {
            "bad": {"module": "Weekly_Daily_Trials", "files": ["../secret.csv"]}
        }), patch.object(bot, "load_data") as load:
            with self.assertRaises(ValueError):
                bot.generate_report("bad")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
