{#
  Default dbt behaviour concatenates target schema and custom schema, which
  would produce STAGING_MARTS. This project has real STAGING and MARTS
  schemas already created in Snowflake, so use the custom name verbatim.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
