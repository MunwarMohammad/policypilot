"""Explicit provisioning command; creates billable infrastructure only when run."""

import argparse


def main():
    from databricks.vector_search.client import VectorSearchClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--create-endpoint", action="store_true")
    args = parser.parse_args()
    client = VectorSearchClient()
    if args.create_endpoint:
        client.create_endpoint(name=args.endpoint, endpoint_type="STANDARD")
    client.wait_for_endpoint(args.endpoint)
    index = client.create_delta_sync_index(
        endpoint_name=args.endpoint,
        index_name=args.index,
        source_table_name=args.table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_dimension=384,
        embedding_vector_column="embedding",
    )
    index.wait_until_ready()
    print("Index ready. After later Delta updates, call get_index(...).sync() and wait for sync.")


if __name__ == "__main__":
    main()
