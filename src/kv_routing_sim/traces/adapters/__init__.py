"""One adapter per input format. Each is a generator yielding `NormalizedRequest`. 📐 designed.

This is one of the project's two extension points (the other is `policies/`), and it is the one
that separates this simulator from Vidur and Frontier: both replay their own trace formats. An
adapter is a single file, which is what makes "add a trace adapter" an honest good-first-issue.
"""
