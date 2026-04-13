# BIRD EX Evaluation Report Template

## 1. Experiment Meta

- Date:
- Environment (Web IDE / Local):
- Code commit / branch:
- Base model:
- EA model:
- Decode params (`temperature`, `max_new_token`, `total_token`, `depth`, `top_k`):
- Eval params (`timeout_sec`, `ignore_row_order`):

## 2. Core Metrics

- Total predictions:
- Matched questions:
- DB found:
- Gold executable:
- Pred executable:
- EX denominator:
- EX matches:
- Execution Accuracy (EX):
- Pred executable rate:

## 3. Failure Breakdown

- `db_not_found`:
- `question_not_found`:
- `gold_sql_empty`:
- `pred_empty_sql`:
- `pred_syntax_error`:
- `pred_runtime_error`:
- `pred_timeout`:
- `exec_mismatch`:

## 4. Top Failure Cases

1. question_id:
   db_id:
   reason:
   pred_sql:
   gold_sql:
   notes:
2. question_id:
   db_id:
   reason:
   pred_sql:
   gold_sql:
   notes:

## 5. Analysis and Next Actions

- Main bottleneck hypothesis:
- Potential fix 1:
- Potential fix 2:
- Next experiment plan:
