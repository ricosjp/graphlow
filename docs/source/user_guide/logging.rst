Logging (optional)
==================

``graphlow`` uses the standard library :mod:`logging` module. Library code in
modules such as ``graphlow.core.mesh`` is wired for logging,
but **no handlers are attached when you import** ``graphlow``.
That keeps library behavior aligned with the usual logging rule.

When to use :func:`graphlow.configure_logging`
----------------------------------------------

:func:`~graphlow.configure_logging` is a **small convenience helper** for
situations where you want readable ``graphlow`` output **without** writing a
full logging config:

- Jupyter notebooks and interactive sessions
- Python REPL
- Short one-off scripts while debugging

It loads the bundled ``logging.conf`` (or a path you pass in) and sets up
handlers for the ``graphlow`` logger. It does **not** run on import so you call
it explicitly when you want that behavior.

When **not** to rely on it
---------------------------

For production services, libraries you ship, or any project with a defined
logging strategy, configure :mod:`logging` yourself.
Do not treat :func:`~graphlow.configure_logging` as
the backbone of your application's logging. It is only a shortcut for
interactive and exploratory use.

See also
--------

- API: :func:`graphlow.configure_logging` in the :doc:`/api_reference/index`
