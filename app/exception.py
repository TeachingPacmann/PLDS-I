# from enum import Enum
from typing import Any, Dict
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

async def starlette_exception_handler(request, exc: StarletteHTTPException):
    response = {
        "status": str(exc.detail),
        "message": str(exc.detail),
        "data" : None
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        response: Dict[str, Any] = {}
        response["status"] = "BAD_REQUEST"
        response["message"] = ", ".join(
             [f"{x['loc'][-1]} - {x['msg']} - {x['type']}" for x in exc.errors()]
        )
        response["data"] = None

        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content=response,
        )

    except Exception as exc:
        raise BaseException(
            message=f"There's an error in the validation handler - {str(exc)}"
        )

class BaseException(Exception):
    message = "INTERNAL SERVER ERROR"
    status_code = 500

    def __init__(self, message: str):
        self.message = message

    def base_return(self) -> Dict:
        return {
            "status": "INTERNAL SERVER ERROR",
            "message": self.message,
            "data" : None
        }