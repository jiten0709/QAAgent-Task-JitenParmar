// playwright.config.js
module.exports = {
    testDir: './src/tests/generated',
    timeout: 30000,
    expect: {
        timeout: 5000
    },
    use: {
        headless: false,
        viewport: { width: 1280, height: 720 },
        screenshot: 'only-on-failure',
        video: 'retain-on-failure',
        trace: 'on-first-retry'
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] }
        },
        {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] }
        },
        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] }
        },
        {
            name: 'Mobile Chrome',
            use: { ...devices['Pixel 5'] }
        }
    ]
};