{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree: .
   :template: autosummary/attribute.rst

{% for item in attributes %}
{% if item not in inherited_members %}
   ~{{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}

{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree: .
   :template: autosummary/method.rst

{% for item in methods %}
{% if item not in inherited_members and item != "__init__" %}
   ~{{ objname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}