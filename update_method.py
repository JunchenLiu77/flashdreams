import re

filepath = "docs/source/developer_guides/new_integration.rst"

with open(filepath, "r") as f:
    content = f.read()

# "Adding a New Model Integration" -> "Adding a New Method"
content = content.replace("Adding a New Model Integration", "Adding a New Method")

# "adding a new integration" -> "adding a new method"
content = content.replace("adding a new integration", "adding a new method")

# "maintain your integration" -> "maintain your method"
content = content.replace("maintain your integration", "maintain your method")

# "your new integration" -> "your new method"
content = content.replace("your new integration", "your new method")

# "an integration typically defines" -> "a method typically defines"
content = content.replace(
    "an integration typically defines", "a method typically defines"
)

# "my integration" -> "my method"
content = content.replace("my integration", "my method")

# "register the integration" -> "register the method"
content = content.replace("register the integration", "register the method")

# "developing a new integration" -> "developing a new method"
content = content.replace("developing a new integration", "developing a new method")

# "documenting your new integration" -> "documenting your new method"
content = content.replace(
    "documenting your new integration", "documenting your new method"
)

with open(filepath, "w") as f:
    f.write(content)

print("Replacements done.")
