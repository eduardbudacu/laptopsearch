# laptopsearch

I've created this dashboard to allow quick exploration of the laptop market on major e-commerce websites from Romania.

It helped see at a glance what are the main options, understand the price ranges and compare between different websites.

The data is scraped using the browser extension https://webscraper.io/ 

![Dashboard](./docs/screenshot.png)

## Deployment on heroku


1. Build container

```
heroku container:push web -a laptopsearchro
```

2. Release

```
heroku container:release web -a laptopsearchro
```

3. Check logs 

```
heroku logs --tail -a laptopsearchro
```