
// OpenGL_Test1View.cpp: COpenGLTest1View 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "OpenGL_Test1.h"
#endif

#include "OpenGL_Test1Doc.h"
#include "OpenGL_Test1View.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// COpenGLTest1View

IMPLEMENT_DYNCREATE(COpenGLTest1View, CView)

BEGIN_MESSAGE_MAP(COpenGLTest1View, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &COpenGLTest1View::OnFilePrintPreview)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
END_MESSAGE_MAP()

// COpenGLTest1View 생성/소멸

COpenGLTest1View::COpenGLTest1View() noexcept
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

COpenGLTest1View::~COpenGLTest1View()
{
}

BOOL COpenGLTest1View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// COpenGLTest1View 그리기

void COpenGLTest1View::OnDraw(CDC* /*pDC*/)
{
	COpenGLTest1Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
}


// COpenGLTest1View 인쇄


void COpenGLTest1View::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL COpenGLTest1View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void COpenGLTest1View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void COpenGLTest1View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}

void COpenGLTest1View::OnRButtonUp(UINT /* nFlags */, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void COpenGLTest1View::OnContextMenu(CWnd* /* pWnd */, CPoint point)
{
#ifndef SHARED_HANDLERS
	theApp.GetContextMenuManager()->ShowPopupMenu(IDR_POPUP_EDIT, point.x, point.y, this, TRUE);
#endif
}


// COpenGLTest1View 진단

#ifdef _DEBUG
void COpenGLTest1View::AssertValid() const
{
	CView::AssertValid();
}

void COpenGLTest1View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

COpenGLTest1Doc* COpenGLTest1View::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(COpenGLTest1Doc)));
	return (COpenGLTest1Doc*)m_pDocument;
}
#endif //_DEBUG


// COpenGLTest1View 메시지 처리기
