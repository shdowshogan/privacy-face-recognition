import argparse

from privacy_store import PrivacyStore


def parse_args():
    parser = argparse.ArgumentParser(description="Revoke user consent and hard-delete embeddings.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--db", default="face_pipeline/privacy.db")
    return parser.parse_args()


def main():
    args = parse_args()
    store = PrivacyStore(db_path=args.db)
    store.revoke_consent(args.user_id)
    print(f"Revoked consent and deleted embeddings for user: {args.user_id}")


if __name__ == "__main__":
    main()
