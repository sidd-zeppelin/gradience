from gradience.autograd.context import Context

def test_context_repr():
    ctx = Context()
    ctx.save_for_backward(1, 2)
    assert repr(ctx) == "Context(saved_tensors=2)"
