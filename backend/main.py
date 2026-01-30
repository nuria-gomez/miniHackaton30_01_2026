import os
from datetime import datetime, date, timedelta
from typing import List, Optional
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Date, ForeignKey, Boolean, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel
from faker import Faker
from dotenv import load_dotenv

# Load environment variables (only in development)
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://nuria:123456@localhost:5434/nuria_extras")
# CRITICAL: Convert postgres:// to postgresql:// for SQLAlchemy compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# CORS configuration - pydantic-settings interpreta List[str] como JSON
# Por eso usamos str con valores separados por comas
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:5173")

def get_cors_origins_list() -> list[str]:
    """Convert comma-separated CORS origins to list"""
    if not CORS_ORIGINS_STR:
        return ["http://localhost:5173"]
    return [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== SQLAlchemy Models ====================

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article_products = relationship("ArticleProduct", back_populates="product")
    sales = relationship("Sale", back_populates="product")


class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    image_url = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=False)
    clicks = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article_products = relationship("ArticleProduct", back_populates="article")
    feedback = relationship("Feedback", back_populates="article")
    social_stats = relationship("SocialStats", back_populates="article")


class ArticleProduct(Base):
    __tablename__ = "article_products"
    
    article_id = Column(Integer, ForeignKey("articles.id"), primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), primary_key=True)
    
    # Relationships
    article = relationship("Article", back_populates="article_products")
    product = relationship("Product", back_populates="article_products")


class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    like = Column(Boolean, nullable=False)
    will_implement = Column(Boolean, nullable=False)
    doctor_email = Column(String(255), nullable=True)
    time_spent = Column(Integer, nullable=False, default=0)  # Time in seconds
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article = relationship("Article", back_populates="feedback")


class Sale(Base):
    __tablename__ = "sales"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    month = Column(String(7), nullable=False)  # Format: "YYYY-MM"
    units_sold = Column(Integer, nullable=False)
    revenue = Column(DECIMAL(10, 2), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="sales")


class SocialStats(Base):
    __tablename__ = "social_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    platform = Column(String(50), nullable=False)  # twitter, instagram, facebook
    positive_comments = Column(Integer, default=0)
    negative_comments = Column(Integer, default=0)
    neutral_comments = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    article = relationship("Article", back_populates="social_stats")


# Create tables
Base.metadata.create_all(bind=engine)

# ==================== Pydantic Schemas ====================

class ProductResponse(BaseModel):
    id: int
    name: str
    category: str


class ArticleResponse(BaseModel):
    id: int
    title: str
    content: str
    image_url: Optional[str]
    published_at: str
    clicks: int
    product_ids: List[int]
    summary: Optional[dict] = None


class ArticleDetailResponse(BaseModel):
    id: int
    title: str
    content: str
    image_url: Optional[str]
    published_at: str
    clicks: int
    products: List[ProductResponse]


class FeedbackCreate(BaseModel):
    article_id: int
    like: bool
    will_implement: bool
    doctor_email: Optional[str] = None
    time_spent: Optional[int] = 0  # Time in seconds


class FeedbackResponse(BaseModel):
    id: int
    article_id: int
    like: bool
    will_implement: bool
    time_spent: int
    submitted_at: str


class SocialStatsResponse(BaseModel):
    platform: str
    positive_comments: int
    negative_comments: int
    neutral_comments: int


class SalesDataResponse(BaseModel):
    month: str
    total_revenue: float
    total_units: int


class FeedbackSummaryResponse(BaseModel):
    likes: int
    dislikes: int
    will_implement: int
    wont_implement: int
    total: int
    clicks: int
    avg_time_spent: int  # Average time in seconds


class ArticleStatsResponse(BaseModel):
    feedback: FeedbackSummaryResponse
    social: List[SocialStatsResponse]
    sales: List[SalesDataResponse]


class DashboardOverviewResponse(BaseModel):
    total_articles: int
    total_feedback: int
    positive_feedback_rate: float
    total_sales_impact: float


# ==================== FastAPI Application ====================

app = FastAPI(title="Bayer Press Team Impact Dashboard", version="1.0.0")

# CRITICAL: Configure CORS with explicit origins from environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==================== Mock Data Initialization ====================

def init_mock_data(db: Session, force: bool = False):
    """Initialize database with mock data"""
    try:
        # 1. Check existing data
        existing_count = db.query(Product).count()
        if existing_count > 0 and not force:
            return {"message": "Database already seeded", "force": False}
        
        # 2. CRITICAL: Delete in reverse dependency order if force
        if force:
            db.query(Feedback).delete()
            db.query(SocialStats).delete()
            db.query(Sale).delete()
            db.query(ArticleProduct).delete()
            db.query(Article).delete()
            db.query(Product).delete()
            db.commit()
        
        # 3. Seed products
        fake = Faker()
        Faker.seed(42)
        
        product_categories = ["Cardiovascular", "Oncology", "Immunology", "Neurology", "Diabetes"]
        products = []
        for i in range(10):
            product = Product(
                name=f"{fake.word().capitalize()} {fake.random_int(100, 999)}",
                category=fake.random_element(product_categories)
            )
            db.add(product)
            products.append(product)
        
        db.commit()
        
        # 4. Seed articles with relationships
        articles = []
        for i in range(15):
            published_at = fake.date_time_between(start_date='-1y', end_date='now')
            article = Article(
                title=fake.sentence(nb_words=8),
                content="\n\n".join(fake.paragraphs(5)),
                image_url=f"https://picsum.photos/seed/{i}/800/400",
                published_at=published_at,
                clicks=fake.random_int(50, 5000)
            )
            db.add(article)
            db.flush()  # Get article.id
            articles.append(article)
            
            # Link to products
            linked_products = fake.random_elements(products, length=fake.random_int(1, 2), unique=True)
            for product in linked_products:
                ap = ArticleProduct(article_id=article.id, product_id=product.id)
                db.add(ap)
            
            # Create social stats
            for platform in ['twitter', 'instagram', 'facebook']:
                social = SocialStats(
                    article_id=article.id,
                    platform=platform,
                    positive_comments=fake.random_int(10, 500),
                    negative_comments=fake.random_int(0, 100),
                    neutral_comments=fake.random_int(5, 200)
                )
                db.add(social)
            
            # Create feedback
            for _ in range(fake.random_int(3, 10)):
                feedback = Feedback(
                    article_id=article.id,
                    like=fake.boolean(),
                    will_implement=fake.boolean(),
                    doctor_email=fake.email() if fake.boolean() else None,
                    time_spent=fake.random_int(60, 600),  # 1-10 minutes in seconds
                    submitted_at=fake.date_time_between(start_date=published_at, end_date='now')
                )
                db.add(feedback)
        
        # 5. Seed sales (12 months per product - last 12 months from now)
        today = datetime.now()
        for product in products:
            for month_offset in range(12):
                # Generate sales for the last 12 months (going backwards)
                month_date = datetime(today.year, today.month, 1) - timedelta(days=30 * month_offset)
                month_str = month_date.strftime("%Y-%m")
                sale = Sale(
                    product_id=product.id,
                    month=month_str,
                    units_sold=fake.random_int(1000, 10000),
                    revenue=Decimal(str(fake.random_int(50000, 500000)))
                )
                db.add(sale)
        
        db.commit()
        
        return {
            "message": "Database seeded successfully",
            "force": force,
            "counts": {
                "products": len(products),
                "articles": len(articles),
                "feedback": db.query(Feedback).count(),
                "social_stats": db.query(SocialStats).count(),
                "sales": db.query(Sale).count()
            }
        }
    
    except Exception as e:
        db.rollback()
        import traceback
        error_msg = f"Error initializing mock data: {str(e)}"
        error_traceback = traceback.format_exc()
        print(error_msg)
        print(error_traceback)
        raise HTTPException(status_code=500, detail=f"{error_msg}. Check server logs for details.")


# ==================== Startup Event ====================

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        init_mock_data(db, force=False)
    finally:
        db.close()


# ==================== API Endpoints ====================

@app.get("/health")
def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/admin/reseed")
def reseed_database(db: Session = Depends(get_db)):
    """Force reseed the database with fresh mock data (admin only)"""
    result = init_mock_data(db, force=True)
    return result


@app.get("/api/articles", response_model=List[ArticleResponse])
def get_articles(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get list of articles with pagination"""
    articles = db.query(Article).offset(skip).limit(limit).all()
    result = []
    for article in articles:
        # Calculate summary stats
        total_feedback = len(article.feedback)
        likes = sum(1 for f in article.feedback if f.like)
        positive_rate = (likes / total_feedback * 100) if total_feedback > 0 else 0
        
        article_dict = {
            "id": article.id,
            "title": article.title,
            "content": article.content,
            "image_url": article.image_url,
            # CRITICAL: Manual date conversion
            "published_at": article.published_at.isoformat() if isinstance(article.published_at, (date, datetime)) else article.published_at,
            "clicks": article.clicks,
            "product_ids": [ap.product_id for ap in article.article_products],
            "summary": {
                "feedback_count": total_feedback,
                "positive_rate": round(positive_rate, 1)
            }
        }
        result.append(ArticleResponse(**article_dict))
    return result


@app.get("/api/articles/{article_id}", response_model=ArticleDetailResponse)
def get_article(article_id: int, db: Session = Depends(get_db)):
    """Get single article with full details"""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    products = [ProductResponse(id=ap.product.id, name=ap.product.name, category=ap.product.category) 
                for ap in article.article_products]
    
    article_dict = {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "image_url": article.image_url,
        # CRITICAL: Manual date conversion
        "published_at": article.published_at.isoformat() if isinstance(article.published_at, (date, datetime)) else article.published_at,
        "clicks": article.clicks,
        "products": products
    }
    return ArticleDetailResponse(**article_dict)


@app.get("/api/articles/{article_id}/stats", response_model=ArticleStatsResponse)
def get_article_stats(article_id: int, db: Session = Depends(get_db)):
    """Get detailed statistics for an article"""
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Feedback summary
    feedback_list = article.feedback
    total_feedback = len(feedback_list)
    likes = sum(1 for f in feedback_list if f.like)
    dislikes = total_feedback - likes
    will_implement = sum(1 for f in feedback_list if f.will_implement)
    wont_implement = total_feedback - will_implement
    
    # Calculate average time spent
    total_time = sum(f.time_spent for f in feedback_list)
    avg_time = int(total_time / total_feedback) if total_feedback > 0 else 0
    
    feedback_summary = FeedbackSummaryResponse(
        likes=likes,
        dislikes=dislikes,
        will_implement=will_implement,
        wont_implement=wont_implement,
        total=total_feedback,
        clicks=article.clicks,
        avg_time_spent=avg_time
    )
    
    # Social stats
    social_list = [
        SocialStatsResponse(
            platform=s.platform,
            positive_comments=s.positive_comments,
            negative_comments=s.negative_comments,
            neutral_comments=s.neutral_comments
        )
        for s in article.social_stats
    ]
    
    # Sales data (aggregate by month for linked products)
    product_ids = [ap.product_id for ap in article.article_products]
    sales_query = db.query(Sale).filter(Sale.product_id.in_(product_ids)).all()
    
    # Group by month
    sales_by_month = {}
    for sale in sales_query:
        month = sale.month
        if month not in sales_by_month:
            sales_by_month[month] = {"revenue": 0, "units": 0}
        sales_by_month[month]["revenue"] += float(sale.revenue)
        sales_by_month[month]["units"] += sale.units_sold
    
    sales_list = [
        SalesDataResponse(
            month=month,
            total_revenue=data["revenue"],
            total_units=data["units"]
        )
        for month, data in sorted(sales_by_month.items())
    ]
    
    return ArticleStatsResponse(
        feedback=feedback_summary,
        social=social_list,
        sales=sales_list
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    """Submit feedback for an article"""
    # Verify article exists
    article = db.query(Article).filter(Article.id == feedback.article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Create feedback
    new_feedback = Feedback(
        article_id=feedback.article_id,
        like=feedback.like,
        will_implement=feedback.will_implement,
        doctor_email=feedback.doctor_email,
        time_spent=feedback.time_spent or 0
    )
    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)
    
    feedback_dict = {
        "id": new_feedback.id,
        "article_id": new_feedback.article_id,
        "like": new_feedback.like,
        "will_implement": new_feedback.will_implement,
        "time_spent": new_feedback.time_spent,
        # CRITICAL: Manual datetime conversion
        "submitted_at": new_feedback.submitted_at.isoformat() if isinstance(new_feedback.submitted_at, datetime) else new_feedback.submitted_at
    }
    return FeedbackResponse(**feedback_dict)


@app.get("/api/products", response_model=List[ProductResponse])
def get_products(db: Session = Depends(get_db)):
    """Get all products"""
    products = db.query(Product).all()
    return [ProductResponse(id=p.id, name=p.name, category=p.category) for p in products]


@app.get("/api/dashboard/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(db: Session = Depends(get_db)):
    """Get aggregate dashboard statistics"""
    total_articles = db.query(Article).count()
    total_feedback = db.query(Feedback).count()
    
    # Calculate positive feedback rate
    likes = db.query(Feedback).filter(Feedback.like == True).count()
    positive_rate = (likes / total_feedback * 100) if total_feedback > 0 else 0
    
    # Calculate total sales impact
    total_sales = db.query(Sale).all()
    total_revenue = sum(float(sale.revenue) for sale in total_sales)
    
    return DashboardOverviewResponse(
        total_articles=total_articles,
        total_feedback=total_feedback,
        positive_feedback_rate=round(positive_rate, 1),
        total_sales_impact=round(total_revenue, 2)
    )


@app.post("/api/seed")
def seed_database(force: bool = False, db: Session = Depends(get_db)):
    """Seed database with mock data (no authentication for MVP)"""
    result = init_mock_data(db, force=force)
    return result


# ==================== Static File Serving (Production) ====================

import os.path

# Mount static files if dist folder exists
if os.path.exists("dist/assets"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

# Root endpoint - serve index.html
@app.get("/")
async def serve_root():
    """Serve the main React app"""
    if os.path.exists("dist/index.html"):
        return FileResponse("dist/index.html")
    return {"error": "Frontend not found. Build required."}

# Catch-all route for SPA (must be at the end, after all API routes)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve React SPA for all non-API routes"""
    # Skip API routes (they're handled above)
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # If dist doesn't exist, return error message
    if not os.path.exists("dist"):
        return {"error": "Frontend not built. Run build script first."}
    
    # If file exists in dist, serve it
    file_path = os.path.join("dist", full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # For root path or any other path, serve index.html (SPA routing)
    index_path = "dist/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {"error": "Frontend index.html not found"}
