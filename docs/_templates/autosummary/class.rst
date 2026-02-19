{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   {% block methods %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: .
      :nosignatures:

   {% for item in methods %}
      {{ objname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: .
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}