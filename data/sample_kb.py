"""Sample knowledge base data for testing."""

SAMPLE_KB_DATA = [
    {
        "title": "How to reset password",
        "content": """To reset your password, follow these steps:
1. Go to the login page and click 'Forgot Password'
2. Enter your email address
3. Check your email for a password reset link
4. Click the link and create a new password
5. Log in with your new password

If you don't receive the email within 5 minutes, check your spam folder or contact support.""",
        "metadata": {"category": "account", "topic": "password"}
    },
    {
        "title": "Return policy",
        "content": """Our return policy allows you to return items within 30 days of purchase. 
Items must be unused and in original packaging. 
To initiate a return:
1. Log into your account
2. Go to Order History
3. Select the order and click 'Return Item'
4. Choose a reason and submit

Refunds are processed within 5-7 business days after we receive the item.""",
        "metadata": {"category": "orders", "topic": "returns"}
    },
    {
        "title": "Shipping information",
        "content": """We offer several shipping options:
- Standard shipping: 5-7 business days (Free on orders over $50)
- Express shipping: 2-3 business days ($15)
- Overnight shipping: Next business day ($30)

You'll receive a tracking number via email once your order ships.
Orders are processed within 1-2 business days.""",
        "metadata": {"category": "orders", "topic": "shipping"}
    },
    {
        "title": "Product warranty",
        "content": """All our products come with a 1-year manufacturer warranty covering defects in materials and workmanship.
The warranty does not cover:
- Normal wear and tear
- Accidental damage
- Unauthorized modifications

To make a warranty claim, contact our support team with your order number and photos of the issue.""",
        "metadata": {"category": "product", "topic": "warranty"}
    },
    {
        "title": "Cancel subscription",
        "content": """To cancel your subscription:
1. Log into your account
2. Go to Account Settings > Subscriptions
3. Click on the active subscription
4. Select 'Cancel Subscription'
5. Confirm cancellation

Your subscription will remain active until the end of the current billing period.
No refunds are provided for partial periods.""",
        "metadata": {"category": "billing", "topic": "subscription"}
    },
    {
        "title": "Update payment method",
        "content": """To update your payment method:
1. Log into your account
2. Go to Account Settings > Payment Methods
3. Click 'Add Payment Method' or edit an existing one
4. Enter your new card details
5. Set as default if desired

All payment information is securely encrypted and PCI compliant.""",
        "metadata": {"category": "billing", "topic": "payment"}
    },
    {
        "title": "Account security tips",
        "content": """Keep your account secure by following these best practices:
- Use a strong, unique password
- Enable two-factor authentication
- Never share your password
- Log out on shared devices
- Review account activity regularly
- Update your password every 90 days
- Be cautious of phishing emails

If you notice suspicious activity, change your password immediately and contact support.""",
        "metadata": {"category": "account", "topic": "security"}
    },
    {
        "title": "Contact support",
        "content": """You can reach our support team in several ways:
- Email: support@example.com (24-48 hour response time)
- Phone: 1-800-123-4567 (Mon-Fri 9am-6pm EST)
- Live chat: Available on our website (Mon-Fri 9am-6pm EST)
- Help Center: help.example.com for self-service articles

For urgent issues, please call our support line.""",
        "metadata": {"category": "support", "topic": "contact"}
    },
]


def get_sample_data():
    """Get sample KB data."""
    return SAMPLE_KB_DATA
