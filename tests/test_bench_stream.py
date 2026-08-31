def test_shared_prefix_is_deterministic_for_a_seed(bench_stream):
    a = bench_stream.make_shared_prefix(64, seed=1234)
    b = bench_stream.make_shared_prefix(64, seed=1234)
    assert a == b
    assert a != ""


def test_shared_prefix_changes_with_the_seed(bench_stream):
    assert bench_stream.make_shared_prefix(64, 1) != bench_stream.make_shared_prefix(64, 2)


def test_shared_prefix_grows_with_the_token_target(bench_stream):
    short = bench_stream.make_shared_prefix(16, 1234)
    long = bench_stream.make_shared_prefix(256, 1234)
    assert len(long) > len(short)


def test_zero_tokens_yields_no_prefix(bench_stream):
    assert bench_stream.make_shared_prefix(0, 1234) == ""


def test_prompts_share_the_prefix_but_differ_in_suffix(bench_stream):
    prefix = bench_stream.make_shared_prefix(64, 1234)
    p0 = bench_stream.build_prompt(prefix, pass_idx=1, req_idx=0)
    p1 = bench_stream.build_prompt(prefix, pass_idx=1, req_idx=1)
    assert p0.startswith(prefix)
    assert p1.startswith(prefix)
    assert p0 != p1


def test_a_later_pass_reuses_the_prefix_with_a_new_suffix(bench_stream):
    prefix = bench_stream.make_shared_prefix(64, 1234)
    first = bench_stream.build_prompt(prefix, 1, 0)
    second = bench_stream.build_prompt(prefix, 2, 0)
    assert second.startswith(prefix)
    assert first != second


def test_no_shared_prefix_falls_back_to_the_fixed_prompt(bench_stream):
    assert bench_stream.build_prompt("", 1, 0) == bench_stream.PROMPT
    assert bench_stream.build_prompt("", 3, 7) == bench_stream.PROMPT


def test_build_body_requests_streaming_with_usage(bench_stream):
    body = bench_stream.build_body("hello", max_tokens=8, no_think=False)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 8
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_no_think_prepends_a_system_message(bench_stream):
    body = bench_stream.build_body("hello", 8, no_think=True)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1] == {"role": "user", "content": "hello"}
