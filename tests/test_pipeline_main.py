import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main as pipeline_main
from src import pipeline_prepare
from src.embedding.vector_store import ingest_chunks_to_store


class _FakeCollection:
    def __init__(self, ids):
        self.ids = set(ids)
        self.deleted = []

    def get(self, include=None):
        return {"ids": sorted(self.ids)}

    def delete(self, ids):
        self.deleted.extend(ids)
        self.ids.difference_update(ids)


class _FakeStore:
    def __init__(self, ids):
        self.collection = _FakeCollection(ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def upsert_chunks(self, chunks):
        self.collection.ids.update(chunk["chunk_id"] for chunk in chunks)
        return len(chunks)

    def count(self):
        return len(self.collection.ids)


class TestIngestPipeline(unittest.TestCase):
    def test_ingest_prunes_stale_ids_after_upsert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "doc_chunks.json"
            chunks_path.write_text(
                json.dumps(
                    [
                        {"chunk_id": "keep", "text": "a", "metadata": {}},
                        {"chunk_id": "new", "text": "b", "metadata": {}},
                    ]
                ),
                encoding="utf-8",
            )
            store = _FakeStore(["keep", "stale"])

            with patch(
                "src.embedding.vector_store.ChromaVectorStore.from_config",
                return_value=store,
            ):
                total = ingest_chunks_to_store(tmpdir, prune=True)

            self.assertEqual(total, 2)
            self.assertEqual(store.collection.ids, {"keep", "new"})
            self.assertEqual(store.collection.deleted, ["stale"])

    def test_ingest_rejects_duplicate_ids_across_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("a_chunks.json", "b_chunks.json"):
                (Path(tmpdir) / name).write_text(
                    json.dumps([{"chunk_id": "duplicate", "text": "x"}]),
                    encoding="utf-8",
                )

            store = _FakeStore([])
            with patch(
                "src.embedding.vector_store.ChromaVectorStore.from_config",
                return_value=store,
            ):
                with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
                    ingest_chunks_to_store(tmpdir, prune=True)


class TestMainOrchestration(unittest.TestCase):
    def test_crawl_failure_stops_before_downstream_stages(self):
        session = MagicMock()
        documents = [{"id": "doc_1"}, {"id": "doc_2"}]

        with (
            patch("src.pipeline_prepare.requests.Session", return_value=session),
            patch("src.pipeline_prepare.time.sleep"),
            patch.object(
                pipeline_prepare.official_crawler,
                "OFFICIAL_DOCUMENTS",
                documents,
            ),
            patch.object(
                pipeline_prepare.official_crawler,
                "crawl_official_document",
                side_effect=[True, False],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "doc_2"):
                pipeline_prepare.crawl_documents()

        session.close.assert_called_once_with()

    @patch("main.ingest_chunks_to_store", return_value=12)
    @patch("main.prepare_data")
    def test_run_pipeline_embedding_only(self, prepare_data, ingest):
        args = argparse.Namespace(
            skip_crawl=False,
            embedding_only=True,
            config=Path("config.yaml"),
            no_prune=False,
        )

        total = pipeline_main.run_pipeline(args)

        self.assertEqual(total, 12)
        prepare_data.assert_called_once_with(skip_crawl=False, embedding_only=True)
        ingest.assert_called_once_with(
            pipeline_main.CHUNKS_DIR,
            config_path=Path("config.yaml"),
            project_root=pipeline_main.ROOT,
            prune=True,
        )


if __name__ == "__main__":
    unittest.main()
