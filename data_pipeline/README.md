# Module 1 — Data Pipeline (`/data_pipeline`)

This module scrapes product data from [books.toscrape.com](https://books.toscrape.com), cleans it, converts prices to INR, stores it in a normalized SQLite database, and queries the database with SQL and with pandas.

| File | What it is |
|---|---|
| `books_pipeline.ipynb` | The whole pipeline in one ordered notebook: scrape → clean → convert → store → query |
| `data/raw_books.csv` | Untouched raw scrape (created by the notebook) |
| `data/books.db` | The SQLite database (created by the notebook; the notebook is also its exact recreation script) |
| `outputs/query_results.md` | Every SQL query string with its full output (created by the notebook) |

## How to run

```bash
cd data_pipeline
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace books_pipeline.ipynb
```
(Or open the notebook in Jupyter and use *Run All*.) It needs internet access to books.toscrape.com and takes about a minute. Running it again rebuilds everything from scratch.

## Scope
**All books in 4 categories:** Mystery, Historical Fiction, Fantasy and Poetry, following every “next” page. Together these hold about 125 books, comfortably above the ≥ 60 books / ≥ 3 categories requirement. Category URLs are read from the site's sidebar, not hard-coded. The notebook asserts the minimum scope.

For each book it captures `title` (from the link's `title` attribute, because the visible text is truncated), `price` (GBP text), `star_rating` (text, e.g. `"Three"`), `availability` (text) and `category`.

## Cleaning decisions

| Raw | Clean column | Rule |
|---|---|---|
| `"£51.77"` | `price_gbp` (float) | Regex extracts the number. The page is decoded as UTF-8 so `£` does not turn into `Â£`. |
| `"Three"` | `rating` (int 1–5) | One…Five → 1…5 |
| `"In stock"` | `in_stock` (bool) | `in stock` → True, `out of stock` → False |

When a field cannot be parsed, the pipeline does not crash. Instead:
- **Numeric fields (`price_gbp`, `rating`) → median imputation.** The median of the same category is used, falling back to the overall median. One bad field should not throw away an otherwise valid product, and the median is not distorted by very cheap or expensive books. Imputed ratings are rounded so they stay whole stars, and an `imputed` flag keeps these rows traceable.
- **`in_stock` → drop the row.** It is a yes/no fact, a boolean has no meaningful median, and a guess would misstate availability.
- **Missing title or category → drop the row**, because the row cannot be identified or linked to a category.
- **Duplicates** (same title in the same category) are dropped.

The notebook shows this working on a handful of deliberately broken test rows before cleaning the real data.

## Currency conversion
`price_inr = price_gbp × 105.50`, using the project's fixed baseline rate **1 GBP = 105.50 INR**. This is an artificial, project-defined constant, not a live or historical market rate, so no API call and no date are involved.

## Database schema (normalized, PK/FK)

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);
CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),   -- SQLite has no boolean type
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```
Each category name is stored once, and books point to it by `category_id`. `PRAGMA foreign_keys = ON` is set, and the notebook runs `PRAGMA foreign_key_check` after loading.

## Queries
All six queries and their full outputs are saved in [`outputs/query_results.md`](outputs/query_results.md) and printed in the notebook.

| # | Question | Clauses |
|---|---|---|
| Q1 | In-stock books under £20 | SELECT, WHERE (run with a raw `sqlite3` cursor) |
| Q2 | 10 most expensive books | ORDER BY, LIMIT |
| Q3 | Distinct star ratings | DISTINCT |
| Q4 | 4–5★ books priced £20–£30 | BETWEEN, IN |
| Q5 | 4+★ books with category name | **JOIN** |
| Q6 | Per-category summary | JOIN, GROUP BY, aggregates |

Q2–Q6 are read into DataFrames with `pd.read_sql`. **Q5 is also rebuilt without SQL** using `pd.merge` on the in-memory `books_df` and `categories_df`. The two results are shown side by side and checked cell by cell with `pd.testing.assert_frame_equal`, which confirms they match.
