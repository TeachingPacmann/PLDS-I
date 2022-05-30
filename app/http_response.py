from fastapi.responses import JSONResponse

def ok(data=None, message=None):
    return JSONResponse(
        status_code=200,
        content={
            "status" : "OK",
            "message" : message,
            "data" : data
        }
    )
    